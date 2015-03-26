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
import cPickle as pickle

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


class Scene(object):
    '''
    stores voxel grid, also labels and perhaps the video of depth images
    will probably have methods to do segmentation etc
    '''
    def __init__(self):
        self.gt_tsdf = None
        self.im_tsdf = None
        self.im = None

    def load_sequence(self, sequence, frame_nos, segment_with_gt, segment=True, save_grids=False, load_implicit=False):
        '''
        loads a sequence of images, the associated gt voxel grid,
        carves the visible tsdf from the images, does segmentation
        '''

        self.sequence = sequence

        # load in the ground truth grid for this scene, and converting nans
        vox_location = paths.RenderedData.ground_truth_voxels(sequence['scene'])
        self.gt_tsdf = voxel_data.load_voxels(vox_location)
        self.gt_tsdf.V[np.isnan(self.gt_tsdf.V)] = -parameters.RenderedVoxelGrid.mu
        self.gt_tsdf.set_origin(self.gt_tsdf.origin)

        # loading in the image
        sequence_frames = sequence['frames'][frame_nos]

        frame_data = paths.RenderedData.load_scene_data(
            sequence['scene'], sequence_frames)

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

        # load in the implicit prediction...
        if load_implicit:
            with open(paths.RenderedData.implicit_prediction_dir % sequence['name'] + 'prediction.pkl') as f:
                self.implicit_tsdf = pickle.load(f)

        '''
        HERE SEGMENTING USING THE GROUND TRUTH
        Want:
            - gt_labels
            - gt_labels_separate
            - gt_tsdf_separate
        Don't need to bother labelling the image using the GT, as this isn't
        needed
        '''

        if segment and segment_with_gt:

            self.gt_labels = self._segment_tsdf_project_2d(
                self.gt_tsdf, z_threshold=2, floor_height=4)

            self.gt_labels_separate = \
                self._separate_binary_grids(self.gt_labels.V, True)

            self.gt_tsdf_separate = \
                self._label_grids_to_tsdf_grids(self.gt_tsdf, self.gt_labels_separate)

            self.gt_im_label = self.im.label_from_grid(self.gt_labels)

        '''
        HERE SEGMENTING USING JUST WHAT IS VISIBLE FROM THE INPUT IMAGE
        Want:
            - visible_labels  # original labels, extended using the projection back into the camera image
            - visible_labels_separate  # This will be projected down and separated into separate binary grids
            - visible_tsdf_separate     # separate tsdf for each label

            - image_labels ???
        '''

        if segment:
            # segmenting with just the visible voxels
            self.visible_labels = self._segment_tsdf_project_2d(
                self.im_tsdf, z_threshold=2, floor_height=4)

            # #### >> transfer the labels from the voxel grid to the image
            self.visible_im_label = self.im.label_from_grid(self.visible_labels)

            # expanding these labels to also cover all the unobserved regions
            uv, to_project_idxs = self.im_tsdf.project_unobserved_voxels(self.im)
            inside_image = self.im.find_points_inside_image(uv)

            # labels of all the non-nan voxels inside the image...
            vox_labels = self.visible_im_label[
                uv[inside_image, 1], uv[inside_image, 0]]

            # now propograte these labels back to the main grid
            temp = to_project_idxs[inside_image]
            self.visible_labels.V.ravel()[temp] = vox_labels
            # << ####

            print "Separate out the partial tsdf into different 'layers' in a grid..."

            self.visible_labels_separate = \
                self._separate_binary_grids(self.visible_labels.V, True)

            temp = self.im_tsdf.copy()
            temp.V[np.isnan(temp.V)] = np.nanmin(temp.V)

            self.visible_tsdf_separate = \
                self._label_grids_to_tsdf_grids(temp, self.visible_labels_separate)


        # # transfer the labels from the voxel grid tothe image
        # self.im.label_from_grid(self.gt_labels)

        # temp = self.im_tsdf.copy()
        # temp.V[np.isnan(temp.V)] = np.nanmin(temp.V)


        if save_grids:

            # save this as a voxel grid...
            savepath = paths.RenderedData.voxlet_prediction_path_short % \
                ('partial_tsdf', sequence['name'])
            self.im_tsdf.save(savepath)

            savepath = paths.RenderedData.voxlet_prediction_path_short % \
                ('visible_voxels', sequence['name'])
            rendersavepath = paths.RenderedData.voxlet_prediction_img_path % \
                ('visible_voxels', sequence['name'])
            self.im_visible.save(savepath)
            self.im_visible.render_view(rendersavepath)

    def extract_single_voxlet(self, index, extract_from, post_transform=None):
        '''
        Helper function to extract shoeboxes from specified locations in voxel
        grid.
        post_transform is a function which is applied to each shoebox
        after extraction
        In this code I am assuming that the voxel grid and image have
        both got label attributes.
        '''
        world_xyz = self.im.get_world_xyz()
        world_norms = self.im.get_world_normals()

        # convert to linear idx
        point_idx = index[0] * self.im.mask.shape[1] + index[1]

        shoebox = voxel_data.ShoeBox(parameters.Voxlet.shape, np.float32)
        shoebox.V *= 0
        shoebox.V += parameters.RenderedVoxelGrid.mu  # set the outside area to -mu
        shoebox.set_p_from_grid_origin(parameters.Voxlet.centre)  # m
        shoebox.set_voxel_size(parameters.Voxlet.size)  # m

        start_x = world_xyz[point_idx, 0]
        start_y = world_xyz[point_idx, 1]

        if parameters.Voxlet.tall_voxlets:
            start_z = parameters.Voxlet.tall_voxlet_height
        else:
            start_z = world_xyz[point_idx, 2]

        shoebox.initialise_from_point_and_normal(
            np.array([start_x, start_y, start_z]),
            world_norms[point_idx],
            np.array([0, 0, 1]))

        #pickle.dump(shoebox, open('/tmp/sbox.pkl', 'w'), protocol=pickle.HIGHEST_PROTOCOL)

        # getting a copy of the voxelgrid, in which only the specified label exists
        if extract_from == 'gt_tsdf':

            this_point_label = self.gt_im_label[index[0], index[1]]
            temp_vgrid = self.gt_tsdf_separate[this_point_label]
            shoebox.fill_from_grid(temp_vgrid)

        elif extract_from == 'visible_tsdf':

            this_point_label = self.visible_im_label[index[0], index[1]]
            temp_vgrid = self.visible_tsdf_separate[this_point_label]
            #print "WARNING - in scene - not using separate grids at the moment..."
            shoebox.fill_from_grid(temp_vgrid)

        elif extract_from == 'implicit_tsdf':

            shoebox.fill_from_grid(self.implicit_tsdf)

        else:
            raise Exception("Don't know how to extract from %s" % extract_from)


        # convert the indices to world xyz space
        if post_transform:
            return post_transform(shoebox)
        else:
            return shoebox

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

    def get_visible_frustrum(self):
        '''
        returns a boolean voxel grid with ones where the voxel is in the
        frustrum of any of the cameras, and zeros otherwise...
        Warning - just doing for a single image, not for a video!
        '''
        carver = carving.VoxelAccumulator()
        carver.set_voxel_grid(self.gt_tsdf.blank_copy())
        inside, _, _ = carver.project_voxels(self.im)
        return inside

    def santity_render(self, save_folder):
        '''
        renders slices though the channels of the scenes
        '''
        u, v = 2, 4

        # Slice though the gt tsdf
        plt.subplot(u, v, 1)
        plt.imshow(self.im.rgb)
        plt.axis('off')
        plt.title('Input RGB image')

        plt.subplot(u, v, 2)
        plt.imshow(self.gt_tsdf.V[:, :, 15])
        plt.axis('off')
        plt.title('Ground truth voxel grid')

        # Slice through the visible labels
        plt.subplot(u, v, 3)
        plt.imshow(self.visible_labels.V[:, :, 15])
        plt.axis('off')
        plt.title('Visible Labels')

        # Slice through the gt labels
        plt.subplot(u, v, 4)
        plt.imshow(self.gt_labels.V[:, :, 15])
        plt.axis('off')
        plt.title('Segmentation of GT grid')

        # Slice though the visible tsdf
        plt.subplot(u, v, 6)
        temp = self.im_tsdf.copy()
        temp.V[np.isnan(temp.V)] = np.nanmin(temp.V)
        plt.imshow(temp.V[:, :, 15])
        plt.axis('off')
        plt.title('Visible TSDF')

        # Image
        plt.subplot(u, v, 7)
        plt.imshow(self.visible_im_label)
        plt.axis('off')
        plt.title('Visible image labels')

        # Image
        plt.subplot(u, v, 8)
        plt.imshow(self.gt_im_label)
        plt.axis('off')
        plt.title('GT Image labels')

        #

        # for idx in [5, 6, 7, 8, 9, 10, 11]:
        #     plt.subplot(u, v, idx)
        #     temp = self.label_grid_tsdf[idx-5]
        #     temp.V[np.isnan(temp.V)] = np.nanmin(temp.V)
        #     plt.imshow(temp.V[:, :, 15])
        #     #label_grid_tsdf

        plt.savefig(save_folder + '/' + self.sequence['name'] + '.png')

    def _segment_tsdf_project_2d(self, tsdf, z_threshold, floor_height):
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
        xy_proj = np.sum(tsdf.V[:, :, floor_height:] < 0, axis=2) > z_threshold
        xy_proj = binary_erosion(xy_proj)
        labels = skimage.measure.label(xy_proj).astype(np.int16)

        # dilate each of the labels
        el = disk(3)
        for idx in range(1, labels.max()+1):
            labels[binary_dilation(labels == idx, el)] = idx

        self.temp_2d_labels = labels

        # propagate these labels back the to the voxels...
        labels3d = np.expand_dims(labels, axis=2)
        labels3d = np.tile(labels3d, (1, 1, tsdf.V.shape[2]))
        labels3d[tsdf.V > parameters.RenderedVoxelGrid.mu] = 0

        labels_3d_grid = tsdf.copy()
        labels_3d_grid.V = labels3d
        labels_3d_grid.V[:, :, :floor_height] = 0

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

        return tsdf_grids

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
