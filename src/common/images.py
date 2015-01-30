'''
classes etc for dealing with depth images
'''
import matplotlib.pyplot as plt
import numpy as np
import scipy.misc
import scipy.ndimage
import h5py
import cv2
from copy import deepcopy
import yaml
import os
import time

import paths
import mesh
import features


class RGBDImage(object):

    def __init__(self, dictionary=[]):
        self.rgb = np.array([])
        self.depth = np.array([])

        if dictionary:
            self.load_from_dict(dictionary)

    def load_rgb_from_img(self, rgb_path, scale_factor=[]):

        self.rgb = scipy.misc.imread(rgb_path)
        if scale_factor:
            self.rgb = scipy.misc.imresize(self.rgb, scale_factor)
        assert(self.rgb.shape[2] == 3)
        self.assert_depth_rgb_equal()

    def load_depth_from_img(self, depth_path):
        self._clear_cache
        self.depth = scipy.misc.imread(depth_path)
        self.assert_depth_rgb_equal()

    def load_depth_from_h5(self, depth_path):
        self._clear_cache
        f = h5py.File(depth_path, 'r')
        self.depth = np.array(f['depth']).astype(np.float32) / 10000
        self.depth[self.depth == 0] = np.nan
        self.assert_depth_rgb_equal()

    def load_depth_from_pgm(self, pgm_path):
        ''' the kinfu routine hack i made does pgm like this'''
        self._clear_cache
        # reading header
        f = open(pgm_path, 'r')
        assert(f.readline().strip() == "P2")
        sizes = f.readline().split()
        height = int(sizes[1])
        width = int(sizes[0])
        max_depth = f.readline()

        # pre-allocating
        self.depth = np.zeros((height, width))
        for row_idx, row in enumerate(f):
            for col_idx, col in enumerate(row.strip().split(" ")):
                self.depth[row_idx, col_idx] = float(col)

        self.depth[self.depth == 0] = np.nan
        f.close()

    def load_kinect_defaults(self):
        '''
        sets the intrinsics to some kinect default
        '''
        K = np.array([[570.0, 0, 320.0], [0, 570.0, 240.0], [0, 0, 1]])
        self.set_intrinsics(K)

    def assert_depth_rgb_equal(self):
        if self.depth.size > 0 and self.rgb.size > 0:
            assert(self.rgb.shape[0] == self.depth.shape[0])
            assert(self.rgb.shape[1] == self.depth.shape[1])

    def disp_channels(self):
        '''plots both the depth and rgb next to each other'''
        plt.clf()
        plt.subplot(121)
        plt.imshow(self.rgb)
        plt.subplot(122)
        plt.imshow(self.depth)
        plt.show()

    def print_info(self):
        '''prints info about the thing'''

        if self.rgb.size > 0:
            print "RGB image has shape: " + str(self.rgb.shape)
        else:
            print "No RGB image present"

        if self.depth.size > 0:
            print "Depth image has shape: " + str(self.depth.shape)
        else:
            print "No Depth image present"

        if hasattr(self, 'mask'):
            print "Mask has shape: " + str(self.mask.shape)

    def compute_edges_and_angles(self, edge_threshold=0.5):
        '''
        computes edges in some manner...
        uses a simple method. Real images should overwrite this function
        to use a more complex edge and angle detection
        '''
        temp_depth = np.copy(self.depth)
        temp_depth[np.isnan(temp_depth)] = 10.0
        # import pdb; pdb.set_trace()

        Ix = cv2.Sobel(temp_depth, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=3)
        Iy = cv2.Sobel(temp_depth, ddepth=cv2.CV_32F, dx=0, dy=1, ksize=3)

        self.angles = np.rad2deg(np.arctan2(Iy, Ix))
        self.angles[np.isnan(self.depth)] = np.nan

        self.edges = np.array((Ix**2 + Iy**2) > edge_threshold**2)
        self.edges = scipy.ndimage.morphology.binary_dilation(self.edges)

    def set_angles_to_zero(self):
        self.angles *= 0

    def reproject_3d(self):
        '''
        creates an (nxm)x3 matrix of all the 3D locations of the points.
        '''
        h, w = self.depth.shape
        us, vs = np.meshgrid(np.arange(w), np.arange(h))
        x = 1000*np.vstack((us.flatten()*self.depth.flatten(),
                            vs.flatten()*self.depth.flatten(),
                            self.depth.flatten()))

        self.xyz = np.dot(self.cam.inv_K, x)
        return self.xyz

    def get_uvd(self):
        '''
        returns (nxm)x3 matrix of all the u, v coordinates and the depth at
        each one
        '''
        h, w = self.depth.shape
        us, vs = np.meshgrid(np.arange(w), np.arange(h))
        return np.vstack((us.flatten(),
                       vs.flatten(),
                       self.depth.flatten())).T

    def set_camera(self, cam_in):
        self.cam = cam_in

    def get_world_xyz(self):
        has_cached = hasattr(self, '_cached_world_xyz')
        if not has_cached:
            self._cached_world_xyz = \
                self.cam.inv_project_points(self.get_uvd())
        return self._cached_world_xyz

    def get_world_normals(self):
        has_cached = hasattr(self, '_cached_world_normals')
        if not has_cached:
            self._cached_world_normals = \
                self.cam.inv_transform_normals(self.normals)
        return self._cached_world_normals

    def _clear_cache(self):
        del self._cached_world_normals
        del self._cached_world_xyz

    def compute_ray_image(self):
        '''
        the ray image is an image where the values represent the
        distance along the rays, as opposed to perpendicular
        '''
        self.reproject_3d()
        dists = np.sqrt(np.sum(self.xyz**2, axis=0))
        return np.reshape(dists, self.depth.shape)/1000

    def set_intrinsics(self, K):
        self.K = K
        self.inv_K = np.linalg.inv(K)

    def disp_channels(self):
        '''plots both the depth and rgb and mask next to each other'''

        plt.clf()
        plt.subplot(221)
        plt.imshow(self.rgb)
        plt.subplot(222)
        plt.imshow(self.depth)
        plt.subplot(223)
        plt.imshow(self.mask)
        plt.subplot(224)
        if hasattr(self, 'edges') and self.edges:
            plt.imshow(self.edges)
            plt.show()

    @classmethod
    def load_from_dict(cls, scene_folder, dictionary):
        '''
        loads an image as defined in 'dictionary', containing entries
        describing where the data is stored, camera parameters etc.
        The scene_folder is the path on the computer to where the folder
        structure is
        '''
        im = cls()
        depth_image_path = os.path.join(scene_folder, dictionary['image'])
        im.load_depth_from_img(depth_image_path)

        # scaling im depth - unsure where I should put this!!
        im.depth = im.depth.astype(float)
        im.depth *= dictionary['depth_scaling']
        im.depth /= 2**16

        # forming the mask
        im.mask = np.abs(im.depth - dictionary['depth_scaling']) < 1e-4

        # setting the camera intrinsics and extrinsics
        extrinsics = \
            np.linalg.inv(np.array(dictionary['pose']).reshape((4, 4)))
        intrinsics = np.array(dictionary['intrinsics']).reshape((3, 3))

        cam = mesh.Camera()
        cam.set_extrinsics(extrinsics)
        cam.set_intrinsics(intrinsics)
        im.set_camera(cam)

        mask_image_path = \
            scene_folder + '/images/mask_%s.png' % dictionary['id']
        # only load the mask if it exists...
        if os.path.exists(mask_image_path):
            im.mask = scipy.misc.imread(mask_image_path) == 255

        # setting the frame id
        im.frame_id = dictionary['id']
        #im.frame_number = count
        return im

    def random_sample_from_mask(self, num_samples):
        '''
        returns indices into the mask
        '''
        indices = np.array(np.nonzero(self.mask)).T
        samples = np.random.randint(0, indices.shape[0], num_samples)
        return indices[samples, :]


class RGBDVideo():
    '''
    stores and loads a sequence of RGBD frames
    '''

    def __init__(self):
        pass
        self.reset()

    def reset(self):
        self.frames = []

    def load_from_yaml(
            self, yamlpath, frames=None):
        '''
        loads a sequence based on a yaml file.
        Image files assumed to be within the specified folder.
        A certain format of YAML is expected. See code for details!!
        '''
        self.reset()
        with open(yamlpath, 'r') as fid:
            frame_data = yaml.load(fid)

        if frames is not None:
            frame_data = [frame_data[frame] for frame in frames]

        # get the folder in which the yaml file resides...
        # (paths in the yaml file are defined relative to this dir)
        folderpath = os.path.dirname(yamlpath)
        self.frames = [RGBDImage.load_from_dict(folderpath, frame)
                        for frame in frame_data]

    def play(self, fps=2.0):
        '''
        plays video in a shitty way.
        Should fix this to be smooth and nice but not sure I can be bothered
        '''
        pause = 1.0 / float(fps)

        plt.ion()
        fig = plt.figure()
        im = plt.imshow(self.frames[0].depth)

        for frame in self.frames:
            im.set_data(frame.depth)
            fig.canvas.draw()
            plt.show()
            time.sleep(pause)

    def subvid(self, frame_numbers):
        '''
        returns a copy of this video comprised of just
        the indicated frame numbers
        '''
        vid_copy = deepcopy(self)
        vid_copy.frames = [self.frames[frame] for frame in frame_numbers]
        return vid_copy

