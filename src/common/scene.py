'''
Class for data about a scene, i.e. a voxel grid plus frames with camera poses
'''
# standard libraries, plotting and system libraries
import numpy as np
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import subprocess as sp
from copy import deepcopy, copy
import time

# IO
import cPickle as pickle
import yaml
from yaml import CLoader
import scipy.misc
import h5py
import scipy.io

# Image processing and machine learning
from skimage.morphology import binary_erosion, binary_dilation, disk
import sklearn.metrics  # for evaluation
import skimage.measure
from scipy.ndimage.interpolation import zoom

# Custom libraries
import voxel_data
import images
import features
import carving
import mesh
import camera


class Scene(object):
    '''
    Stores voxel grid, labels and the video of depth images
    Also has methods to do segmentation etc
    '''
    def __init__(self, mu=None, voxlet_params=None):
        self.gt_tsdf = None
        self.im_tsdf = None
        self.im = None
        self.mu = mu
        self.voxlet_params = voxlet_params

    def _load_scene_data(self, scene_dir, frame_idxs=None):
        '''
        Returns a list of frames from a scene
        If frames then returns only the specified frame numbers
        Opening the yaml file is actually quite surprisingly slow
        '''
        if os.path.exists(scene_dir + '/poses/') and frame_idxs is not None:
            # doing it the quick way! one frame at a time

            def load_frame(idx):
                fname = scene_dir + '/poses/%06d.yaml' % idx
                return yaml.load(open(fname), Loader=CLoader)

            if isinstance(frame_idxs, list):
                frames = [load_frame(idx) for idx in frame_idxs]
            else:
                frames = load_frame(frame_idxs)

        else:
            # this is much slower when poses file is large
            with open(scene_dir + '/poses.yaml', 'r') as f:
                frames = yaml.load(f, Loader=CLoader)

            if frame_idxs is not None:
                if isinstance(frame_idxs, list):
                    frames = [frames[idx] for idx in frame_idxs]
                else:
                    frames = frames[frame_idxs]

        return frames

    def sample_points(self, num_to_sample, sample_grid_size=None, additional_mask=None, nyu=False):
        '''
        Sampling locations at which to extract/place voxlets
        '''
        sample_rate = copy(self.im.depth)

        shape = self.im.depth.shape
        if not nyu:
            sample_rate[self.im.get_world_xyz()[:, 2].reshape(shape) < 0.035] = 0
            sample_rate[np.isnan(self.gt_im_label)] = 0
            sample_rate[self.im.normals[:, 2].reshape(shape) > -0.1] = 0

        # import pdb; pdb.set_trace()
        sample_rate[self.im.mask==0] = 0

        # normals approximately pointing upwards
        sample_rate[self.im.get_world_normals()[:, 2].reshape(shape) > 0.98] = 0

        if sample_rate.sum() == 0:
            raise Exception("Sample rate is zero sum, cannot sample any points")

        scipy.io.savemat('/tmp/samples.mat', {'sr':sample_rate})
        sample_rate = sample_rate.astype(np.float64)
        sample_rate /= sample_rate.sum().astype(np.float64)

        samples = np.random.choice(shape[0]*shape[1], num_to_sample, p=sample_rate.flatten())
        self.sampled_idxs = np.array(np.unravel_index(samples, shape)).T

        return self.sampled_idxs

    def _apply_normalised_homo_transform(self, xyz, trans):
        '''
        applies homogeneous transform, and also does the normalising...
        '''
        temp = self._apply_homo_transformation(xyz, trans)
        return temp[:, :3] / temp[:, 3][:, np.newaxis]

    def _apply_homo_transformation(self, xyz, trans):
        '''
        apply a 4x4 transformation matrix to the vertices
        '''
        n = xyz.shape[0]
        temp = np.concatenate((xyz, np.ones((n, 1))), axis=1).T
        temp_transformed = trans.dot(temp).T
        return temp_transformed

    def load_sequence(self, sequence, frame_nos, segment_with_gt, segment=True,
            save_grids=False, voxel_normals=False, carve=True, segment_base=None,
            original_nyu=False):
        '''
        loads a sequence of images, the associated gt voxel grid,
        carves the visible tsdf from the images, does segmentation
        '''
        self.sequence = sequence

        # load in the ground truth grid for this scene, and converting nans
        voxel_data_path = sequence['folder'] + sequence['scene'] + '/tsdf.dat'
        voxel_meta_path = sequence['folder'] + sequence['scene'] + '/tsdf_meta.yaml'

        self.gt_tsdf = voxel_data.WorldVoxels.load_from_dat(
            voxel_data_path, voxel_meta_path)

        # this i s a nasty hack, which I have to do because I was foolish and carved
        # each set of data with a different offset.
        # I never got round to recarving, and it takes a while...
        # The floor height is the number of voxels which contain floor, starting at
        # the bottom of the voxel grid.
        if sequence['folder'].endswith('_0/'):
            # one bit of traingin data
            floor_height = 5
        elif sequence['folder'].endswith('_1/'):
            floor_height = 7
        elif sequence['folder'].endswith('_2/'):
            # other bit of training data
            floor_height = 7
        else:
            floor_height = 0

        self.floor_height = floor_height

        self.gt_tsdf.V[np.isnan(self.gt_tsdf.V)] = -self.mu
        self.gt_tsdf.set_origin(self.gt_tsdf.origin, self.gt_tsdf.R)

        # loading in the image
        sequence_frames = sequence['frames'][frame_nos]

        self.frame_data = self._load_scene_data(
            sequence['folder'] + sequence['scene'], sequence_frames)

        self.im = images.RGBDImage.load_from_dict(
            sequence['folder'] + sequence['scene'], self.frame_data,
            original_nyu=original_nyu)

        if original_nyu:
            fx_rgb = 5.1885790117450188e+02;
            fy_rgb = 5.1946961112127485e+02;
            cx_rgb = 3.2558244941119034e+02;
            cy_rgb = 2.5373616633400465e+02;
            K = np.array([[fx_rgb, 0, cx_rgb], [0, fy_rgb, cy_rgb], [0, 0, 1]])
            self.im.cam.set_intrinsics(K)

        # while I'm here - might as well save the image as a voxel grid
        if carve:
            video = images.RGBDVideo()
            video.frames = [self.im]
            carver = carving.Fusion()
            carver.set_video(video)
            carver.set_voxel_grid(self.gt_tsdf.blank_copy())
            self.im_tsdf, self.im_visible = carver.fuse(self.mu, inlier_threshold=2)

            # computing normals...
        norm_engine = features.Normals()
        self.im.normals = norm_engine.compute_bilateral_normals(self.im, stepsize=2)

        # update the mask to take into account the normals...
        norm_nans = np.any(np.isnan(self.im.get_world_normals()), axis=1)
        self.im.mask = np.logical_and(
            ~norm_nans.reshape(self.im.mask.shape),
            self.im.mask)

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

            if segment_base:
                temp_tsdf = deepcopy(self.gt_tsdf)
                height_in_vox = int(segment_base / self.gt_tsdf.vox_size)
                temp_tsdf.V = temp_tsdf.V[:, :, height_in_vox:]
            else:
                temp_tsdf = self.gt_tsdf

            self.gt_labels = self._segment_tsdf_project_2d(
                temp_tsdf, floor_height=floor_height, z_threshold=1)

            self.gt_labels_separate = \
                self._separate_binary_grids(self.gt_labels.V, True)

            self.gt_tsdf_separate = \
                self._label_grids_to_tsdf_grids(self.gt_tsdf, self.gt_labels_separate)

            self.gt_im_label = self.im.label_from_grid(self.gt_labels)

        else:
            self.gt_im_label = self.im.mask

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

        start_x = world_xyz[point_idx, 0]
        start_y = world_xyz[point_idx, 1]

        if self.voxlet_params['tall_voxlets']:
            start_z = self.voxlet_params['tall_voxlet_height']
            vox_centre = \
                np.array(self.voxlet_params['shape'][:2]) * \
                self.voxlet_params['size'] * \
                np.array(self.voxlet_params['relative_centre'][:2])
            vox_centre = np.append(vox_centre, start_z)
        else:
            vox_centre = \
                np.array(self.voxlet_params['shape']) * \
                self.voxlet_params['size'] * \
                np.array(self.voxlet_params['relative_centre'])
            start_z = world_xyz[point_idx, 2]

        shoebox.set_p_from_grid_origin(vox_centre)  # m
        shoebox.set_voxel_size(self.voxlet_params['size'])  # m
        shoebox.initialise_from_point_and_normal(
            np.array([start_x, start_y, start_z]),
            world_norms[point_idx],
            np.array([0, 0, 1]))

        #pickle.dump(shoebox, open('/tmp/sbox.pkl', 'w'), protocol=pickle.HIGHEST_PROTOCOL)

        # getting a copy of the voxelgrid, in which only the specified label exists
        if extract_from == 'gt_tsdf':
            this_point_label = self.gt_im_label[index[0], index[1]]
            if np.isnan(this_point_label):
                # this shouldn't happen too much, only due to rounding errors
                print "Nan in sampled point"
                shoebox.fill_from_grid(self.gt_tsdf)
            else:
                temp_vgrid = self.gt_tsdf_separate[this_point_label]
                shoebox.fill_from_grid(temp_vgrid)

        elif extract_from == 'visible_tsdf':
            this_point_label = self.visible_im_label[index[0], index[1]]
            temp_vgrid = self.visible_tsdf_separate[this_point_label]
            # print "WARNING - in scene - not using separate grids at the moment..."
            shoebox.fill_from_grid(temp_vgrid)

        elif extract_from == 'im_tsdf':
            shoebox.fill_from_grid(self.im_tsdf)

        elif extract_from == 'actual_tsdf':
            shoebox.fill_from_grid(self.gt_tsdf)
        else:
            raise Exception("Don't know how to extract from %s" % extract_from)

        # convert the indices to world xyz space
        if post_transform:
            return post_transform(shoebox)
        else:
            return shoebox

    def set_gt_tsdf(self, tsdf_in, floor_height=None):
        self.gt_tsdf = deepcopy(tsdf_in)

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

        plt.savefig(save_folder + '/' + self.sequence['name'] + '.png')

    def _segment_tsdf_project_2d(self, tsdf, z_threshold, floor_height=0):
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
        el = disk(5)
        for idx in range(1, labels.max()+1):
            labels[binary_dilation(labels == idx, el)] = idx

        self.temp_2d_labels = labels
        self.sum_grid = np.sum(tsdf.V[:, :, floor_height:] < 0, axis=2)

        # propagate these labels back the to the voxels...
        labels3d = np.expand_dims(labels, axis=2)
        labels3d = np.tile(labels3d, (1, 1, tsdf.V.shape[2]))

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
            # to_set_to_nan = self.gt_labels.V != idx
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

    def form_evaluation_region(self, extra_mask=None):
        '''
        forms a region of the grid which we should evaluate over
        '''
        temp = np.logical_or(self.im_tsdf.V < 0, np.isnan(self.im_tsdf.V))
        voxels_to_evaluate = np.logical_and(
            temp, self.get_visible_frustrum().reshape(temp.shape))

        floor_t = self.gt_tsdf.blank_copy()
        floor_t.V[:, :, :6] = 1
        voxels_to_evaluate = np.logical_and(floor_t.V == 0, voxels_to_evaluate)

        if extra_mask is not None:
            voxels_to_evaluate = np.logical_and(extra_mask, voxels_to_evaluate)

        return voxels_to_evaluate

    def evaluate_prediction(self, V, voxels_to_evaluate=None, extra_mask=None):
        '''
        evalutes a prediction grid, assuming to be the same size and position
        as the ground truth grid...
        '''
        assert(V.shape[0] == self.gt_tsdf.V.shape[0])
        assert(V.shape[1] == self.gt_tsdf.V.shape[1])
        assert(V.shape[2] == self.gt_tsdf.V.shape[2])

        # deciding which voxels to evaluate over...
        if voxels_to_evaluate is None:
            voxels_to_evaluate = self.form_evaluation_region(extra_mask)
            self.voxels_to_evaluate = voxels_to_evaluate

        # getting the ground truth TSDF voxels
        gt = self.gt_tsdf.V[voxels_to_evaluate] < 0
        if np.isnan(gt).sum() > 0:
            raise Exception('Oops, should not be nans here')

        # Getting the relevant predictions
        V_to_eval = V[voxels_to_evaluate]
        V_to_eval[np.isnan(V_to_eval)] = +self.mu
        prediction = V_to_eval < 0

        # now doing IOU
        union = np.logical_or(gt, prediction)
        intersection = np.logical_and(gt, prediction)

        tp = float(np.logical_and(gt, prediction).sum())
        tn = float(np.logical_and(~gt, ~prediction).sum())
        fp = float(np.logical_and(~gt, prediction).sum())
        fn = float(np.logical_and(gt, ~prediction).sum())

        # Now doing the final evaluation
        results = {}
        results['iou'] = float(intersection.sum()) / float(union.sum())
        # tic = time.time()
        # results['auc'] = float(sklearn.metrics.roc_auc_score(gt, V_to_eval))
        # print "AUC", time.time() - tic

        if (tp + fp) > 0:
            results['precision'] = tp / (tp + fp)
        else:
            results['precision'] = 0

        if (tp + fn) > 0:
            results['recall'] = tp / (tp + fn)
        else:
            results['recall'] = 0

        return results

    def render_visible(self, savepath, xy_centre=True, keep_obj=False,
            save_obj=True, actually_render=True, flip=True):

        H, W = self.im.depth.shape

        # idxs exclude the bottom row
        idxs = np.arange(self.im.depth.size - W)

        # and remove the right column
        right_col = np.mod(idxs+1, W) == 0

        # remove triangles with too big a jump
        gradx, grady = np.gradient(self.im.depth[:-1, :])
        grad = np.sqrt(gradx**2 + grady**2)
        jumps = grad > 0.05
        jumps = binary_dilation(jumps)

        # rmeove edges
        edges = self.im.depth.copy()[:-1, :] * 0
        edges[:15, :] = 1
        edges[-15:, :] = 1
        edges[:, -15:] = 1
        edges[:, :15] = 1

        print right_col.shape, jumps.ravel().shape, edges.ravel().shape
        idxs = idxs[~np.logical_or.reduce(
            (right_col, jumps.ravel(), edges.ravel()==1))]

        # combine all the triangles
        tris = np.vstack([idxs, idxs + W, idxs + W + 1]).T
        tris2 = np.vstack([idxs + W + 1, idxs + 1, idxs]).T
        all_tris = np.vstack((tris, tris2))

        ms = mesh.Mesh()
        ms.faces = all_tris
        ms.vertices = self.im.get_world_xyz()
        ms.remove_nan_vertices()

        if flip:
            ms.vertices[:, 0] *= -1

        # Now do blender render and copy the obj file if needed...
        if xy_centre:
            # T = ms.vertices[:, :2]
            temp = self.gt_tsdf
            cen = temp.origin + (np.array(temp.V.shape) * temp.vox_size) / 2.0
            ms.vertices[:, :2] -= cen[:2]
            ms.vertices[:, 2] -= 0.05
            # ms.vertices *= 1.5

        if save_obj:
            print "Saving to ", savepath + '.obj'
            ms.write_to_obj(savepath + '.obj')

        blend_path = os.path.expanduser('~/projects/shape_sharing/src/'
            'rendered_scenes/spinaround/spin.blend')
        blend_py_path = os.path.expanduser('~/projects/shape_sharing/src/'
            'rendered_scenes/spinaround/blender_spinaround_frame.py')

        if actually_render:
            subenv = os.environ.copy()
            subenv['BLENDERSAVEFILE'] = savepath
            sp.call(['blender',
                     blend_path,
                     "-b", "-P",
                     blend_py_path],
                     env=subenv,
                     stdout=open(os.devnull, 'w'),
                     close_fds=True)

        if not keep_obj:
            os.remove(savepath + '.obj')

        return ms
