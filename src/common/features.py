import numpy as np
import scipy.io
from scipy.linalg import svd
from sklearn.neighbors import KDTree
from copy import copy, deepcopy
import carving

class CobwebEngine(object):
    '''
    A different type of patch engine, only looking at points in the compass directions
    '''
    def __init__(self, t, fixed_patch_size=False, use_mask=None):
        # self.t is the stepsize at a depth of 1 m
        self.t = float(t)
        self.fixed_patch_size = fixed_patch_size
        self.use_mask = use_mask

    def set_image(self, im):
        self.im = im
        self.depth = copy(self.im.depth)
        if self.use_mask:
            self.depth[im.mask==0] = np.nan

    def get_cobweb(self, index):
        '''extracts cobweb for a single index point'''
        row, col = index

        start_angle = 0
        start_depth = self.im.depth[row, col]

        focal_length = self.im.cam.estimate_focal_length()
        if self.fixed_patch_size:
            offset_dist = focal_length * self.t
        else:
            offset_dist = (focal_length * self.t) / start_depth

        # computing all the offsets and angles efficiently
        offsets = offset_dist * np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        rad_angles = np.deg2rad(start_angle + np.array(range(0, 360, 45)))

        rows_to_take = (float(row) - np.outer(offsets, np.sin(rad_angles))).astype(int).flatten()
        cols_to_take = (float(col) + np.outer(offsets, np.cos(rad_angles))).astype(int).flatten()

        # defining the cobweb array ahead of time
        fv_length = rows_to_take.shape[0]  # == cols_to_take.shape[0]
        cobweb = np.nan * np.zeros((fv_length, )).flatten()

        # working out which indices are within the image bounds
        to_use = np.logical_and.reduce((rows_to_take >= 0,
                                        rows_to_take < self.depth.shape[0],
                                        cols_to_take >= 0,
                                        cols_to_take < self.depth.shape[1]))
        rows_to_take = rows_to_take[to_use]
        cols_to_take = cols_to_take[to_use]

        # computing the diff vals and slotting into the correct place in the cobweb feature
        vals = self.depth[rows_to_take, cols_to_take] - self.depth[row, col]
        cobweb[to_use] = vals
        self.rows, self.cols = rows_to_take, cols_to_take

        return np.copy(cobweb.flatten())

    def extract_patches(self, indices):
        return [self.get_cobweb(index) for index in indices]


class Normals(object):
    '''
    A python 'normals' class.
    Contains a few different ways of computing normals of a depth image
    '''
    def __init__(self):
        pass

    def normalize_v3(self, arr):
        '''
        Normalize a numpy array of 3 component vectors shape=(n,3)
        '''
        lens = np.sqrt(arr[:, 0] ** 2 + arr[:, 1] ** 2 + arr[:, 2] ** 2)
        arr[:, 0] /= lens
        arr[:, 1] /= lens
        arr[:, 2] /= lens
        return arr

    def compute_bilateral_normals(self, im, stepsize=2):
        '''
        wrapper for compute_normals, but does filtering of the image first
        '''
        carver = carving.Fusion()
        im2 = deepcopy(im)
        im2._clear_cache()
        im2.depth = carver._filter_depth(im.depth)
        return self.compute_normals(im2, stepsize)

    def compute_normals(self, im, stepsize=2):
        '''
        one method of computing normals
        '''
        xyz = im.reproject_3d()

        x = xyz[0, :].reshape(im.depth.shape)
        y = xyz[1, :].reshape(im.depth.shape)
        z = xyz[2, :].reshape(im.depth.shape)

        dx0, dx1 = np.gradient(x, stepsize)
        dy0, dy1 = np.gradient(y, stepsize)
        dz0, dz1 = np.gradient(z, stepsize)

        dx0 = dx0.flatten()
        dx1 = dx1.flatten()
        dy0 = dy0.flatten()
        dy1 = dy1.flatten()
        dz0 = dz0.flatten()
        dz1 = dz1.flatten()

        dxyz0 = self.normalize_v3(np.vstack((dx0, dy0, dz0)).T).T
        dxyz1 = self.normalize_v3(np.vstack((dx1, dy1, dz1)).T).T
        cross = np.cross(dxyz0, dxyz1, axis=0)

        return self.normalize_v3(cross.T)

    def compute_curvature(self, im, offset=1):
        '''
        I must have got this code from the internet somewhere,
        but I don't remember where....
        '''
        Z = im.depth

        Zy, Zx  = np.gradient(Z, offset)
        Zxy, Zxx = np.gradient(Zx, offset)
        Zyy, _ = np.gradient(Zy, offset)

        H = (Zx**2 + 1)*Zyy - 2*Zx*Zy*Zxy + (Zy**2 + 1)*Zxx

        H = -H/(2*(Zx**2 + Zy**2 + 1)**(1.5))

        K = (Zxx * Zyy - (Zxy ** 2)) /  (1 + (Zx ** 2) + (Zy **2)) ** 2

        return H, K, Zyy, Zxx

    def voxel_normals(self, im, vgrid):
        '''
        compute the normals from a voxel grid
        '''
        offset = 3
        xyz = im.get_world_xyz()
        inliers = np.ravel(im.mask)

        # padding the array
        t = 10
        pad_width = ((offset+t, offset+t), (offset+t, offset+t), (offset+t, offset+t))
        padded = np.pad(vgrid.V, pad_width, 'edge')
        padded[np.isnan(padded)] = np.nanmin(padded)

        idx = vgrid.world_to_idx(xyz[inliers]) + offset + t
        # print idx
        # print idx.shape
        ds = np.eye(3) * offset

        diffs = []
        for d in ds:
            plus = (idx + d).astype(int)
            minus = (idx - d).astype(int)

            diffs.append(
                padded[plus[:, 0], plus[:, 1], plus[:, 2]] -
                padded[minus[:, 0], minus[:, 1], minus[:, 2]])

        diffs = np.vstack(diffs).astype(np.float32)
        length = np.linalg.norm(diffs, axis=0)
        length[length==0] = 0.0001

        diffs /= length
        # print np.isnan(diffs).sum()
        # print diffs.shape

        # now convert the normals to image space instead of world space...
        image_norms = np.dot(im.cam.inv_H[:3, :3], diffs).T

        # pad out array to the correct size for future computations...
        output_norms = np.zeros((im.mask.size, 3), dtype=np.float32)
        output_norms[inliers, :] = image_norms
        return output_norms


class SampledFeatures(object):
    '''
    Samples values from a voxel grid about a point and a normal
    '''
    def __init__(self, num_rings, radius):
        '''
        units are in real world space I think...
        '''
        self.num_rings = num_rings
        self.radius = radius

    def set_scene(self, sc):
        self.sc = sc

    def _get_sample_locations(self, point, normal):
        '''
        returns a Nx3 array of the *world* locations of where to sample from
        assumes the grid to be orientated correctly with the up direction
        pointing upwards
        '''
        # print "norm is ", normal
        # print "angle is ", start_angle
        start_angle = np.rad2deg(np.arctan2(normal[0], normal[1]))
        ring_offsets = self.radius * (1 + np.arange(self.num_rings))

        # now combining...
        all_locations = []
        for r in ring_offsets:
            for elevation in np.deg2rad(np.array([-45, 0, 45])):
                z = r * np.sin(elevation)
                cos_elevation = np.cos(elevation)
                for azimuth in np.deg2rad(start_angle + np.arange(0, 360, 45)):
                    x = r * np.sin(azimuth) * cos_elevation
                    y = r * np.cos(azimuth) * cos_elevation
                    all_locations.append([x, y, z])

        # add top and bottom locations
        for ring_radius in ring_offsets:
            all_locations.append([0, 0, ring_radius])
            all_locations.append([0, 0, -ring_radius])

        locations = np.array(all_locations)

        # finally add on the start location...
        return locations + point

    def _single_sample(self, point, normal):
        # sampled feature for a single point
        world_sample_location = self._get_sample_locations(point, normal)
        idxs = self.sc.im_tsdf.world_to_idx(world_sample_location)
        sampled_values = self.sc.im_tsdf.get_idxs(idxs, check_bounds=True)

        return sampled_values

    def sample_idx(self, idx):
        # samples at each of the N locations, and returns some shape thing
        point_idx = idx[0] * self.sc.im.mask.shape[1] + idx[1]

        xyz = self.sc.im.get_world_xyz()
        norms = self.sc.im.get_world_normals()
        # print "in sample idx ", norms.shape
        return self._single_sample(xyz[point_idx], norms[point_idx])

    def sample_idxs(self, idxs):
        # samples at each of the N locations, and returns some shape thing
        point_idxs = idxs[:, 0] * self.sc.im.mask.shape[1] + idxs[:, 1]

        xyz = self.sc.im.get_world_xyz()
        norms = self.sc.im.get_world_normals()
        return np.vstack(
            [self._single_sample(xyz[idx], norms[idx]) for idx in point_idxs])
