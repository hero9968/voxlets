'''
Class for data about a scene, i.e. a voxel grid plus frames with camera poses
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
import yaml
from yaml import CLoader
from copy import deepcopy
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import scipy.misc
import mesh
import h5py

from copy import copy
from scipy.ndimage.interpolation import zoom


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

    def load_bigbird_matrices(self, folder, modelname, imagename):
        '''
        loads the extrinsics and intrinsics for a bigbird camera
        camera name is something like 'NP5'
        '''
        cameraname, angle = imagename.split('_')

        # loading the pose and calibration files
        calib = h5py.File(folder + modelname + "/calibration.h5", 'r')
        pose_path = paths.bigbird_folder + modelname + "/poses/NP5_" + angle + "_pose.h5"
        pose = h5py.File(pose_path, 'r')

        # extracting extrinsic and intrinsic matrices
        np5_to_this_camera = np.array(calib['H_' + cameraname + '_from_NP5'])
        mesh_to_np5 = np.linalg.inv(np.array(pose['H_table_from_reference_camera']))

        intrinsics = np.array(calib[cameraname + '_depth_K'])

        # applying to the camera
        # self.set_extrinsics(np5_to_this_camera.dot(mesh_to_np5))
        # self.set_intrinsics(intrinsics)
        return np5_to_this_camera.dot(mesh_to_np5), intrinsics
        calib.close()
        pose.close()


    def load_bb_sequence(self, sequence, voxel_normals=True):
        '''
        load from the bigbird dataset
        '''
        # self.sequence = sequence
        # rgbpath = sequence['folder'] + '/' + sequence['scene'] + '/' + sequence['pose_id'] + '.jpg'
        # depthpath = sequence['folder'] + '/' + sequence['scene'] + '/' + sequence['pose_id'] + '.h5'
        # maskpath = sequence['folder'] + '/' + sequence['scene'] + '/masks/' + sequence['pose_id'] + '_mask.pbm'

        # self.im = images.RGBDImage()
        # self.im.rgb = zoom(scipy.misc.imread(rgbpath), (0.5, 0.5, 1.0))[:480, :, :]
        # # self.im.load_mask_from_pbm()
        # self.im.load_depth_from_h5(depthpath)

        # self.im.load_mask_from_pbm(maskpath, 0.5)
        # self.im.mask = self.im.mask[:480, :]
        # # self.im.mask[np.isnan(self.im.depth)] = 0

        # self.im.cam = mesh.Camera()
        # mats = self.load_bigbird_matrices(sequence['folder'], sequence['scene'], sequence['pose_id'])
        # self.im.cam.set_extrinsics(mats[0])
        # self.im.cam.set_intrinsics(mats[1])


        # print self.im.rgb
        matpath = sequence['folder'] + '/' + sequence['scene'] + '/' + sequence['pose_id'] + '.mat'
        D = scipy.io.loadmat(matpath)
        left, right, top, bottom = D['aabb'][0]
        # print D.keys()
        self.im = images.RGBDImage()
        self.im.depth = np.nan * np.zeros((480, 640))
        self.im.depth[top:bottom+1, left:right+1] = D['orig_d']

        self.im.rgb = D['rgb']
        # np.zeros((480, 640, 3)).astype(np.uint8)
        # self.im.rgb[top:bottom+1, left:right+1] =
        # print self.im.rgb.shape

        # clean up the mask
        thresh = np.median(D['orig_d'].flatten()[D['mask'].flatten()==1]) + 0.2
        temp_mask = np.logical_and.reduce((D['orig_d'] < thresh, D['orig_d'] >0, D['mask']==1))

        self.TT = np.zeros((480, 640)).astype(np.uint8)
        self.TT[top:bottom+1, left:right+1] = D['mask']

        self.im.mask = np.zeros((480, 640))
        # .astype(np.uint8)
        self.im.mask[top:bottom+1, left:right+1] = temp_mask

        self.im.depth[self.im.depth==0] = np.nan
        # + 0.02
        # self.im.rgb = D['rgb']

        self.im.cam = mesh.Camera()
        self.im.cam.load_bigbird_matrices(sequence['folder'], sequence['scene'], sequence['pose_id'])

        # self.im.cam.set_extrinsics(np.linalg.inv(D['T'][0][0][2]).T)

        # self.im.cam.set_intrinsics(D['T'][0][0][1])
        # print D['T'][0][0][1]
        # print D['T'][0][0][2]

        self.gt_im_label = copy(self.im.mask) + 100
        # extr = D['T'][0][0][2].T

        # ne = features.Normals()
        rec_mask = np.zeros((480, 640))
        rec_mask[top:bottom+1, left:right+1] = 1
        self.im.normals = np.zeros((480*640, 3)) * np.nan
        old_shape = [3, D['mask'].shape[1], D['mask'].shape[0]]
        self.im.normals[rec_mask.flatten()==1, :] = D['norms'].T.reshape(old_shape).transpose((0, 2, 1)).reshape((3, -1)).T
        # self.im.mask = np.logical_and(self.im.mask, ~np.isnan(self.im.normals[:, 2].reshape(self.im.depth.shape)))

        # np.any(np.abs(self.im.normals)==1, axis


    def sample_points(self, num_to_sample, sample_grid_size=None, additional_mask=None, nyu=False):
        '''
        Sampling locations at which to extract/place voxlets
        '''
        if False:
            # perhaps add in some kind of thing to make sure that we don't take normals pointing the wrong way
            xyz = self.im.get_world_xyz()

            good_normals = self.im.normals[:, 2].reshape((480, 640)) < -0.5
            if additional_mask:
                additional_mask = np.logical_and(additional_mask, good_normals)
            else:
                additional_mask = good_normals

            # Over-sample the points to begin with
            sampled_idxs = \
                self.im.random_sample_from_mask(4*num_to_sample, additional_mask=additional_mask)
            linear_idxs = sampled_idxs[:, 0] * self.im.mask.shape[1] + \
                sampled_idxs[:, 1]

            # Now I want to choose points according to the x-y grid
            # First, discretise the location of each point
            sampled_xy = xyz[linear_idxs, :2]
            sampled_xy /= sample_grid_size
            sampled_xy = np.round(sampled_xy).astype(int)

            sample_dict = {}
            for idx, row in zip(sampled_idxs, sampled_xy):
                if (row[0], row[1]) in sample_dict:
                    sample_dict[(row[0], row[1])].append(idx)
                else:
                    sample_dict[(row[0], row[1])] = [idx]

            sampled_points = []
            while len(sampled_points) < num_to_sample:

                for key, value in sample_dict.iteritems():
                    # add the top item to the sampled points
                    if len(value) > 0 and len(sampled_points) < num_to_sample:
                        sampled_points.append(value.pop())

            self.sampled_idxs = np.array(sampled_points)
        else:
            sample_rate = copy(self.im.depth)
            print sample_rate.sum()

            shape = self.im.depth.shape
            if not nyu:
                print "Not NYU for some reason"
                sample_rate[self.im.get_world_xyz()[:, 2].reshape(shape) < 0.035] = 0
                sample_rate[np.isnan(self.gt_im_label)] = 0
                sample_rate[self.im.normals[:, 2].reshape(shape) > -0.1] = 0
            # import pdb; pdb.set_trace()
            print "Mask", (self.im.mask!=0).sum()
            print sample_rate.shape, self.im.mask.shape
            sample_rate[self.im.mask==0] = 0
            print sample_rate.sum()

            # normals approximately pointing upwards
            sample_rate[self.im.get_world_normals()[:, 2].reshape(shape) > 0.98] = 0
            print sample_rate.sum()

            if sample_rate.sum() == 0:
                raise Exception("Sample rate is zero sum, cannot sample any points")

            import scipy.io
            scipy.io.savemat('/tmp/samples.mat', {'sr':sample_rate})
            sample_rate /= sample_rate.sum()

            samples = np.random.choice(shape[0]*shape[1], num_to_sample, p=sample_rate.flatten())
            self.sampled_idxs = np.array(np.unravel_index(samples, shape)).T

        return self.sampled_idxs

        # # loading the GT mesh...
        # vox_path = '/media/ssd/data/bigbird_meshes/' + sequence['scene'] + '/meshes/voxelised.vox'
        # self.populate_from_vox_file(idx_path)

        # xyz =  D['xyz'].T.reshape(old_shape).transpose((0, 2, 1)).reshape((3, -1)).T
        # self.im._cached_world_xyz = self._apply_normalised_homo_transform(D['xyz'], np.linalg.inv(extr))
        # print D['xyz'].shape

    # def norrms(self):
    #     # while I'm here - might as well save the image as a voxel grid

    #     video = images.RGBDVideo()
    #     video.frames = [self.im]
    #     carver = carving.Fusion()
    #     carver.set_video(video)
    #     carver.set_voxel_grid(self.gt_tsdf.blank_copy())
        # self.im_tsdf, self.im_visible = carver.fuse(self.mu)
    #     # video = images.RGBDVideo()
    #     # video.frames = [self.im]
    #     # carver = carving.Fusion()
    #     # carver.set_video(video)
    #     # carver.set_voxel_grid(self.gt_tsdf.blank_copy())
    #     # self.im_tsdf, self.im_visible = carver.fuse(self.mu)
    #     # norm_engine = features.Normals()
    #     # self.im.normals = norm_engine.voxel_normals(self.im, self.im_tsdf)

    def populate_from_vox_file(self, filepath):
        '''
        Loads 3d locations from my custom .vox file.
        My .vox file is almost but not quite a yaml
        Seemed that using pyyaml was incredibly slow so did this instead... bit of a hack!2
        '''
        f = open(filepath, 'r')
        f.readline() # origin:
        self.set_origin(np.array(f.readline().split(" ")).astype(float))
        f.readline() # extents:
        f.readline() # extents - don't care about this
        f.readline() # voxel_size:
        self.set_voxel_size(float(f.readline().strip()))
        f.readline() # vox:
        idx = np.array([line.split() for line in f]).astype(int)
        f.close()
        self.init_and_populate(idx)

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
            save_grids=False, voxel_normals=False, carve=True, segment_base=None):
        '''
        loads a sequence of images, the associated gt voxel grid,
        carves the visible tsdf from the images, does segmentation
        '''

        self.sequence = sequence

        # load in the ground truth grid for this scene, and converting nans
        vox_location = sequence['folder'] + sequence['scene'] + \
            '/ground_truth_tsdf.pkl'

        self.set_gt_tsdf(voxel_data.load_voxels(vox_location))

        # this i s a nasty hack, which I have to do because I was foolish and carved
        # each set of data with a different offset.
        # I never got round to recarving, and it takes a while...
        # The floor height is the number of voxels which contain floor, starting at
        # the bottom of the voxel grid.
        if sequence['folder'].endswith('ta/'):
            # one bit of traingin data
            floor_height = 5
        elif sequence['folder'].endswith('ta1/'):
            floor_height = 7
        elif sequence['folder'].endswith('ta2/'):
            # other bit of training data
            floor_height = 7
        else:
            floor_height = None

        self.floor_height = floor_height

        self.gt_tsdf.V[np.isnan(self.gt_tsdf.V)] = -self.mu
        self.gt_tsdf.set_origin(self.gt_tsdf.origin, self.gt_tsdf.R)

        # loading in the image
        sequence_frames = sequence['frames'][frame_nos]

        self.frame_data = self._load_scene_data(
            sequence['folder'] + sequence['scene'], sequence_frames)

        self.im = images.RGBDImage.load_from_dict(
            sequence['folder'] + sequence['scene'], self.frame_data)

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

        # self.im.normals = norm_engine.voxel_normals(self.im, self.im_tsdf)

            # if voxel_normals and voxel_normals == 'im_tsdf':
            #     self.im.normals = norm_engine.voxel_normals(self.im, self.im_tsdf)
            # elif voxel_normals:
            # else:
            #     self.im.normals = norm_engine.compute_normals(self.im)

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

        '''
        HERE SEGMENTING USING JUST WHAT IS VISIBLE FROM THE INPUT IMAGE
        Want:
            - visible_labels  # original labels, extended using the projection back into the camera image
            - visible_labels_separate  # This will be projected down and separated into separate binary grids
            - visible_tsdf_separate     # separate tsdf for each label

            - image_labels ???
        '''

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
            #print "WARNING - in scene - not using separate grids at the moment..."
            shoebox.fill_from_grid(temp_vgrid)

        elif extract_from == 'im_tsdf':
            shoebox.fill_from_grid(self.im_tsdf)

        elif extract_from == 'actual_tsdf':
            shoebox.fill_from_grid(self.gt_tsdf)
            # print np.isnan(shoebox.V).sum()
            # print point_idx, index, self.im.mask.shape
            # print np.array([start_x, start_y, start_z])
            # print world_norms[point_idx]
            # print self.gt_tsdf.world_to_idx(np.array([[start_x, start_y, start_z]]))
            # print ""
        else:
            raise Exception("Don't know how to extract from %s" % extract_from)

        # convert the indices to world xyz space
        if post_transform:
            return post_transform(shoebox)
        else:
            return shoebox

    def set_gt_tsdf(self, tsdf_in, floor_height=None):
        # print "Warning - I think these should be re-commented for testing..."
        # floor_height_in_vox = float(floor_height) / float(tsdf_in.vox_size)
        # if floor_height_in_vox > 15:
        #     raise Exception("This seems excessive")
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

    def evaluate_prediction(self, V):
        '''
        evalutes a prediction grid, assuming to be the same size and position
        as the ground truth grid...
        '''
        assert(V.shape[0] == self.gt_tsdf.V.shape[0])
        assert(V.shape[1] == self.gt_tsdf.V.shape[1])
        assert(V.shape[2] == self.gt_tsdf.V.shape[2])

        # deciding which voxels to evaluate over...
        temp = np.logical_or(self.im_tsdf.V < 0, np.isnan(self.im_tsdf.V))
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
        results['auc'] = float(sklearn.metrics.roc_auc_score(gt, V_to_eval))

        if (tp + fp) > 0:
            results['precision'] = tp / (tp + fp)
        else:
            results['precision'] = 0

        if (tp + fn) > 0:
            results['recall'] = tp / (tp + fn)
        else:
            results['recall'] = 0

        # fpr, tpr, _ = sklearn.metrics.roc_curve(gt, V_to_eval)
        # results['fpr'] = fpr
        # results['tpr'] = tpr

        return results
