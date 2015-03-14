'''
quick class to store data about a SCENE!

'''

import numpy as np
from skimage.morphology import binary_erosion, binary_dilation, disk
import skimage.measure
import paths
import parameters
import voxel_data
import images
import features
import carving

class Scene(object):
    '''
    stores voxel grid, also labels and perhaps the video of depth images
    will probably have methods to do segmentation etc
    '''
    def __init__(self):
        self.gt_tsdf = None
        self.im_tsdf = None
        self.im = None

    def load_sequence(self, sequence, frame_nos, save_grids=False):
        '''
        loads a sequence of images, the associated gt voxel grid,
        carves the visible tsdf from the images, does segmentation
        '''

        # load in the ground truth grid for this scene, and converting nans
        vox_location = paths.RenderedData.ground_truth_voxels(sequence['scene'])
        self.gt_tsdf = voxel_data.load_voxels(vox_location)
        self.gt_tsdf.V[np.isnan(self.gt_tsdf.V)] = -parameters.RenderedVoxelGrid.mu
        self.gt_tsdf.set_origin(self.gt_tsdf.origin)

        # loading in the image
        sequence_frames = sequence['frames'][frame_nos]
        print sequence_frames
        frame_data = paths.RenderedData.load_scene_data(
            sequence['scene'], sequence_frames)
        print frame_data
        self.im = images.RGBDImage.load_from_dict(
            paths.RenderedData.scene_dir(sequence['scene']),
            frame_data)

        # computing normals...
        norm_engine = features.Normals()
        self.im.normals = norm_engine.compute_normals(self.im)

        # while I'm here - might as well save the image as a voxel grid
        video = images.RGBDVideo()
        video.frames = [self.im]
        carver = carving.Fusion()
        carver.set_video(video)
        carver.set_voxel_grid(self.gt_tsdf.blank_copy())
        self.im_tsdf, self.im_visible = \
            carver.fuse(parameters.RenderedVoxelGrid.mu)
        self.labels = \
            self._segment_tsdf_project_2d(z_threshold=2, floor_height=4)

        self.separated_labels = self._separate_binary_grids(self.labels.V, True)
        self.label_grid_tsdf = \
            self._label_grids_to_tsdf_grids(self.im_tsdf, self.separated_labels)

        # # here need to construct the label grids - but this time we do not have
        # # a full voxel grid, so the same segmentation may not work so well
        # labels_grids = {}
        # for idx in np.unique(self.im_tsdf.labels):
        #     temp = self.im_tsdf.copy()
        #     temp.V[self.im_tsdf.labels != idx] = parameters.RenderedVoxelGrid.mu
        #     labels_grids[idx] = temp

        # transfer the labels from the voxel grid tothe image
        self.im.label_from_grid(self.labels)

        # with open('/tmp/partial_segmented.pkl', 'w') as f:
        #     pickle.dump(dict(labels_grids=labels_grids,
        #     partial_tsdf=partial_tsdf, image=im),
        #     f, protocol=pickle.HIGHEST_PROTOCOL)

        if save_grids:

            # save this as a voxel grid...
            savepath = paths.RenderedData.voxlet_prediction_path % \
                ('partial_tsdf', sequence['name'])
            self.im_tsdf.save(savepath)

            savepath = paths.RenderedData.voxlet_prediction_path % \
                ('visible_voxels', sequence['name'])
            rendersavepath = paths.RenderedData.voxlet_prediction_img_path % \
                ('visible_voxels', sequence['name'])
            self.im_visible.save(savepath)
            self.im_visible.render_view(rendersavepath)

    def set_gt_tsdf(self, tsdf_in):
        self.gt_tsdf = tsdf_in

    def set_im_tsdf(self, tsdf_in):
        '''
        for now just store the tsdf for a single image
        In future, may have the tsdf for each image or set of images etc
        '''
        self.im_tsdf = tsdf_in

    def set_im(self, im):
        self.im = im

    def _segment_tsdf_project_2d(self, z_threshold, floor_height):
        '''
        segments a voxel grid by projecting full voxels onto the xy
        plane and segmenting in 2D. Naive but good for testing
        I'm doing this here in the scene class as it makes sense to keep the
        VoxelGrid class clean!

        z_threshold is the number of full voxels in z-dir needed in order
            to be considered a full pixel in xy projection

        floor_height is in voxels
        '''
        # using partial_tsdf as I don't think I should ever be using the
        # GT tsdf? It doesn't really make any sense as I will never have the GT
        # tsdf at training or at test...?
        xy_proj = np.sum(self.im_tsdf.V[:, :, floor_height:] < 0, axis=2) > z_threshold
        xy_proj = binary_erosion(xy_proj)
        labels = skimage.measure.label(xy_proj).astype(np.int16)

        # dilate each of the labels
        el = disk(3)
        for idx in range(1, labels.max()+1):
            labels[binary_dilation(labels == idx, el)] = idx

        self.temp_2d_labels = labels

        # propagate these labels back the to the voxels...
        labels3d = np.expand_dims(labels, axis=2)
        labels3d = np.tile(labels3d, (1, 1, self.im_tsdf.V.shape[2]))
        labels3d[self.im_tsdf.V > parameters.RenderedVoxelGrid.mu] = 0

        labels_3d_grid = self.gt_tsdf.copy()
        labels_3d_grid.V = labels3d

        return labels_3d_grid

    def _label_grids_to_tsdf_grids(self, tsdf, binary_grids):
        '''
        extracts the tsdf corresponding to each of the binary grids
        (Run separate_binary_grids before doing this...)
        '''
        tsdf_grids = {}
        for idx, reg in binary_grids.iteritems():
            temp = tsdf.copy()
            temp.V[~reg] = np.nanmax(tsdf.V)
            tsdf_grids[idx] = temp

        self.tsdf_grids = tsdf_grids
        return self.tsdf_grids

    def _separate_binary_grids(self, vox_labels, flatten_labels_down=True):
        '''
        takes a 3D array of labels, and returns a dict
        where keys are labels and values are a 3D boolean
        array with one where that label is true
        If flatten_labels_down == True, then cumsums along
        the -ve z axis.
        (Run separate_binary_grids after doing this...)
        '''
        each_label_region = {}

        for idx in np.unique(vox_labels):
            label_temp = vox_labels == idx

            if flatten_labels_down:
                label_temp = np.cumsum(
                    label_temp[:, :, ::-1], axis=2)[:, :, ::-1]

            each_label_region[idx] = label_temp > 0

        return each_label_region
