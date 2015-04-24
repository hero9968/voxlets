'''
quick class to store data about a SCENE!

'''

import numpy as np
from skimage.morphology import binary_erosion, binary_dilation, disk
import sklearn.metrics  # for evaluation
import skimage.measure
import voxel_data
import images
import features
import carving
import cPickle as pickle
import paths
import yaml

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


class Scene(object):
    '''
    stores voxel grid, also labels and perhaps the video of depth images
    will probably have methods to do segmentation etc
    '''
    def __init__(self, mu, voxlet_params):
        self.gt_tsdf = None
        self.im_tsdf = None
        self.im = None
        self.mu = mu
        self.voxlet_params = voxlet_params

    def _load_scene_data(self, scene_dir, frame_idxs=None):
        '''
        returns a list of frames from a scene
        if frames then returns only the specified frame numbers
        '''
        with open(scene_dir + '/poses.yaml', 'r') as f:
            frames = yaml.load(f)

        # Using "!= None" because if I don't and frame_idxs==0 then this fails
        if frame_idxs != None:
            if isinstance(frame_idxs, list):
                frames = [frames[idx] for idx in frame_idxs]
            else:
                frames = frames[frame_idxs]

        return frames

    def load_sequence(self, sequence, frame_nos, segment_with_gt, segment=True,
            save_grids=False, load_implicit=False, voxel_normals=False):
        '''
        loads a sequence of images, the associated gt voxel grid,
        carves the visible tsdf from the images, does segmentation
        '''

        self.sequence = sequence

        # load in the ground truth grid for this scene, and converting nans
        vox_location = sequence['folder'] + sequence['scene'] + \
            '/ground_truth_tsdf.pkl'
        if sequence['folder'][-3:-1] == 'ta':
            # we are in the training data - this has less floor
            self.set_gt_tsdf(voxel_data.load_voxels(vox_location), 0.015)
        else:
            self.set_gt_tsdf(voxel_data.load_voxels(vox_location), 0.035)
        print self.gt_tsdf.origin

        self.gt_tsdf.V[np.isnan(self.gt_tsdf.V)] = -self.mu
        self.gt_tsdf.set_origin(self.gt_tsdf.origin,
                self.gt_tsdf.R)

        # loading in the image
        sequence_frames = sequence['frames'][frame_nos]

        frame_data = self._load_scene_data(
            sequence['folder'] + sequence['scene'], sequence_frames)

        self.im = images.RGBDImage.load_from_dict(
            sequence['folder'] + sequence['scene'], frame_data)

        # while I'm here - might as well save the image as a voxel grid
        video = images.RGBDVideo()
        video.frames = [self.im]
        carver = carving.Fusion()
        carver.set_video(video)
        carver.set_voxel_grid(self.gt_tsdf.blank_copy())
        self.im_tsdf, self.im_visible = carver.fuse(self.mu)

        # computing normals...
        norm_engine = features.Normals()
        if voxel_normals and voxel_normals == 'im_tsdf':
            self.im.normals = norm_engine.voxel_normals(self.im, self.im_tsdf)
        elif voxel_normals:
            self.im.normals = norm_engine.voxel_normals(self.im, self.gt_tsdf)
        else:
            self.im.normals = norm_engine.compute_normals(self.im)

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
                self.gt_tsdf, z_threshold=1, floor_height=4)

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

        # if segment:
        #     # segmenting with just the visible voxels
        #     self.visible_labels = self._segment_tsdf_project_2d(
        #         self.im_tsdf, z_threshold=2, floor_height=4)

        #     # #### >> transfer the labels from the voxel grid to the image
        #     self.visible_im_label = self.im.label_from_grid(self.visible_labels)

        #     # expanding these labels to also cover all the unobserved regions
        #     uv, to_project_idxs = self.im_tsdf.project_unobserved_voxels(self.im)
        #     inside_image = self.im.find_points_inside_image(uv)

        #     # labels of all the non-nan voxels inside the image...
        #     vox_labels = self.visible_im_label[
        #         uv[inside_image, 1], uv[inside_image, 0]]

        #     # now propograte these labels back to the main grid
        #     temp = to_project_idxs[inside_image]
        #     self.visible_labels.V.ravel()[temp] = vox_labels
        #     # << ####

        #     print "Separate out the partial tsdf into different 'layers' in a grid..."

        #     self.visible_labels_separate = \
        #         self._separate_binary_grids(self.visible_labels.V, True)

        #     temp = self.im_tsdf.copy()
        #     temp.V[np.isnan(temp.V)] = np.nanmin(temp.V)

        #     self.visible_tsdf_separate = \
        #         self._label_grids_to_tsdf_grids(temp, self.visible_labels_separate)


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

        shoebox = voxel_data.ShoeBox(self.voxlet_params['shape'], np.float32)
        shoebox.V *= np.nan
        # shoebox.V += self.mu  # set the outside area to -mu
        shoebox.set_p_from_grid_origin(self.voxlet_params['centre'])  # m
        shoebox.set_voxel_size(self.voxlet_params['size'])  # m

        start_x = world_xyz[point_idx, 0]
        start_y = world_xyz[point_idx, 1]

        if self.voxlet_params['tall_voxlets']:
            start_z = self.voxlet_params['tall_voxlet_height']
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

        elif extract_from == 'im_tsdf':

            shoebox.fill_from_grid(self.im_tsdf)

        elif extract_from == 'implicit_tsdf':

            shoebox.fill_from_grid(self.implicit_tsdf)

        elif extract_from == 'actual_tsdf':
            shoebox.fill_from_grid(self.gt_tsdf)

        else:
            raise Exception("Don't know how to extract from %s" % extract_from)


        # convert the indices to world xyz space
        if post_transform:
            return post_transform(shoebox)
        else:
            return shoebox

    def set_gt_tsdf(self, tsdf_in, floor_height):
        print "Warning - I think these should be uncommented for training..."
        # floor_height_in_vox = float(floor_height) / float(tsdf_in.vox_size)
        # if floor_height_in_vox > 15:
            # raise Exception("This seems excessive")
        self.gt_tsdf = deepcopy(tsdf_in)
        # self.gt_tsdf.V = tsdf_in.V[:, :, floor_height_in_vox:]
        # self.gt_tsdf.origin[2] += floor_height


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
        labels3d[tsdf.V > self.mu] = 0

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
            to_set_to_nan = np.logical_and(self.gt_labels.V != idx, self.gt_labels.V != 0)
            temp.V[to_set_to_nan] = np.nan
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

    def evaluate_prediction(self, V):
        '''
        evalutes a prediction grid, assuming to be the same size and position
        as the ground truth grid...
        '''
        assert(V.shape[0] == self.gt_tsdf.V.shape[0])
        assert(V.shape[1] == self.gt_tsdf.V.shape[1])
        assert(V.shape[2] == self.gt_tsdf.V.shape[2])

        temp = np.logical_or(self.im_tsdf.V < 0, np.isnan(self.im_tsdf.V))

        # deciding which voxels to evaluate over...
        voxels_to_evaluate = np.logical_and(
            temp, self.get_visible_frustrum().reshape(temp.shape))
        floor_t = self.gt_tsdf.blank_copy()
        floor_t.V[:, :, :6] = 1
        voxels_to_evaluate = np.logical_and(floor_t.V == 0, voxels_to_evaluate)
        self.voxels_to_evaluate = voxels_to_evaluate

        # getting the ground truth TSDF voxels
        gt = self.gt_tsdf.V[voxels_to_evaluate] < 0
        if np.isnan(gt).sum() > 0:
            raise Exception('Oops, should not be nans here')

        # Getting the relevant predictions
        V_to_eval = V[voxels_to_evaluate]
        V_to_eval[np.isnan(V_to_eval)] = +self.mu

        # now doing IOU
        union = np.logical_or(gt, (V_to_eval < 0))
        intersection = np.logical_and(gt, (V_to_eval < 0))
        self.union = union
        self.intersection = intersection

        tp = np.logical_and(gt, V_to_eval < 0).sum()
        tn = np.logical_and(~gt, V_to_eval > 0).sum()
        fp = np.logical_and(~gt, V_to_eval < 0).sum()
        fn = np.logical_and(gt, V_to_eval > 0).sum()


        # Now doing the final evaluation
        results = {}
        results['iou'] = float(intersection.sum()) / float(union.sum())
        results['auc'] = sklearn.metrics.roc_auc_score(gt, V_to_eval)
        results['precision'] = float(tp) / (float(tp) + float(fp))
        # sklearn.metrics.precision_score(gt, V_to_eval < 0)
        results['recall'] = float(tp) / (float(tp) + float(fn))
        # sklearn.metrics.recall_score(gt, V_to_eval < 0)

        fpr, tpr, _ = sklearn.metrics.roc_curve(gt, V_to_eval)
        # results['fpr'] = fpr
        # results['tpr'] = tpr

        return results

