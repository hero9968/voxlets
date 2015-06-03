'''
This is an engine for extracting rotated patches from a depth image.
Each patch is rotated so as to be aligned with the gradient in depth at that point
Patches can be extracted densely or from pre-determined locations
Patches should be able to vary to be constant-size in real-world coordinates
(However, perhaps this should be able to be turned off...)
'''

import numpy as np
import scipy.stats
import scipy.io
from numbers import Number
import scipy.stats as stats
from scipy.spatial import KDTree
from scipy.linalg import svd
from skimage.restoration import denoise_bilateral
from skimage.color import rgb2gray
from scipy.ndimage import uniform_filter
from sklearn.neighbors import KDTree

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib as mpl
from copy import copy, deepcopy

import carving

# helper function related to features...
def replace_nans_with_col_means(X):
    '''
    http://stackoverflow.com/questions/18689235/numpy-array-replace-nan-values-with-average-of-columns
    '''
    col_mean = stats.nanmean(X,axis=0)
    col_mean[np.isnan(col_mean)] = 0
    inds = np.where(np.isnan(X))
    X[inds]=np.take(col_mean,inds[1])
    return X


class CobwebEngine(object):
    '''
    A different type of patch engine, only looking at points in the compass directions
    '''

    def __init__(self, t, fixed_patch_size=False, mask=None):

        # the stepsize at a depth of 1 m
        self.t = float(t)

        # dimension of side of patch in real world 3D coordinates
        #self.input_patch_hww = input_patch_hww

        # if fixed_patch_size is True:
        #   step is always t in input image pixels
        # else:
        #   step varies linearly with depth. t is the size of step at depth of 1.0
        self.fixed_patch_size = fixed_patch_size

        self.mask = mask

    def set_image(self, im):
        self.im = im
        self.depth = copy(self.im.depth)
        if self.mask is not None:
            self.depth[self.mask==0] = np.nan
            self.depth[im.get_world_xyz()[:, 2].reshape(im.depth.shape) < 0.035] = np.nan

    def get_cobweb(self, index):
        '''extracts cobweb for a single index point'''
        row, col = index

        start_angle = 0#self.im.angles[row, col]
        # take the start depth from the full image...
        # all other depths come from whatever the mask says...
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


        #idxs = np.ravel_multi_index((rows_to_take, cols_to_take), dims=self.im.depth.shape, order='C')
        #cobweb = self.im.depth.take(idxs) - self.im.depth[row, col]


class SpiderEngine(object):
    '''
    Engine for computing the spider (compass) features
    '''

    def __init__(self, im):
        '''
        sets the depth image and computes the distance transform
        '''
        self.im = im
        #self.distance_transform = dt.get_compass_images()


    def compute_spider_features(self, idxs):
        '''
        computes the spider feature for a given point
        '''
        return self.im.spider_channels[idxs[:, 0], idxs[:, 1], :]


class PatchPlot(object):
    '''
    Aim of this class is to plot boxes at specified locations, scales and orientations
    on a background image
    '''

    def __init__(self):
        pass

    def set_image(self, image):
        self.im = im
        plt.imshow(im.depth)

    def plot_patch(self, index, angle, width):
        '''plots a single patch'''
        row, col = index
        bottom_left = (col - width/2, row - width/2)
        angle_rad = np.deg2rad(angle)

        # creating patch
        #print bottom_left, width, angle
        p_handle = patches.Rectangle(bottom_left, width, width, color="red", alpha=1.0, edgecolor='r', fill=None)
        transform = mpl.transforms.Affine2D().rotate_around(col, row, angle_rad) + plt.gca().transData
        p_handle.set_transform(transform)

        # adding to current plot
        plt.gca().add_patch(p_handle)

        # plotting line from centre to the edge
        plt.plot([col, col + width * np.cos(angle_rad)],
                 [row, row + width * np.sin(angle_rad)], 'r-')


    def plot_patches(self, indices, scale_factor):
        '''plots the patches on the image'''

        scales = [scale_factor * self.im.depth[index[0], index[1]] for index in indices]

        angles = [self.im.angles[index[0], index[1]] for index in indices]

        plt.hold(True)

        for index, angle, scale in zip(indices, angles, scales):
            self.plot_patch(index, angle, scale)

        plt.hold(False)
        plt.show()


class Normals(object):
    '''
    finally it is here: a python 'normals' class.
    probably will contain a few different ways of computing normals of a depth
    image
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

    def nn_normals(self, im, n):
        '''
        computes normals from nearest neighbours
        unfinished - seems slow
        '''
        xyz = im.reproject_3d().T
        nans = np.any(np.isnan(xyz), axis=1)
        print nans.shape, nans.sum()
        print "xyz is shape", xyz.shape
        dsd
        sub_xyz = xyz[~nans, :]
        tree = KDTree(sub_xyz)
        _, neighbour_idxs = tree.query(sub_xyz, k=20, eps=0.2)

        print neighbour_idxs.shape

        # for each set of neighbours, now do the thing
        for this_neighbour_idxs in neighbour_idxs:

            n_xyz = sub_xyz[this_neighbour_idxs]
            print 'N xyz ', n_xyz.shape

            # todo - can set options to make this quicker
            U, s, Vh = svd(n_xyz)
            print U, s, Vh

        # return diffs.T

class SampledFeatures(object):
    '''
    samples features from a voxel grid
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

    def sample_idxs(self, idxs):
        # samples at each of the N locations, and returns some shape thing
        point_idxs = idxs[:, 0] * self.sc.im.mask.shape[1] + idxs[:, 1]

        xyz = self.sc.im.get_world_xyz()
        norms = self.sc.im.get_world_normals()
        print "in sample idxs ", norms.shape
        return np.vstack(
            [self._single_sample(xyz[idx], norms[idx]) for idx in point_idxs])


class RegionFeatureEngine(object):
    '''
    I separate each feature into its own function for readability and
    debugability
    '''
    def __init__(self, params=None):
        self.params = params

    def set_si_dict(self, si_dict):
        self.si_dict = si_dict

    def compute_features(self, image, mask):

        # extract the 2d and 3d points corresponding to this segment
        xyz = image.get_world_xyz()[mask.ravel()]
        # norms = image.get_world_normals()[mask.ravel()]
        rgb = np.vstack((image.rgb[:, :, a].ravel()[mask.ravel()] for a in [0, 1, 2])).T

        # now compute each of the features
        features = {}
        features['bounding_box'] = self._get_bounding_box(xyz)
        features['hog'] = self._get_hog(image, mask)
        features['rgb_hist'] = self._get_rgb_hist(rgb)
        features['rgb_mean'] = self._get_rgb_mean(rgb)
        features['shape_distribution'] = self._get_shape_dist(xyz)
        # features['spin_image'] = self._get_si_hist(xyz)

        return features

    def _get_si_hist(self, xyz):
        print "Not yet implemented si histogram"
        return None
        # inlier_radius =

        # tree = KDTree(xyz, leaf_size=2)
        # for p in xyz:
        #     tree.

    def _get_hog(self, img, mask):
        '''
        a lot of code taken from skimage
        '''
        n_bins = 8

        grey = np.sqrt(rgb2gray(img.rgb))

        '''First stage - gradients'''
        gx = np.empty(grey.shape, dtype=np.double)
        gx[:, 0] = 0
        gx[:, -1] = 0
        gx[:, 1:-1] = grey[:, 2:] - grey[:, :-2]
        gy = np.empty(grey.shape, dtype=np.double)
        gy[0, :] = 0
        gy[-1, :] = 0
        gy[1:-1, :] = grey[2:, :] - grey[:-2, :]

        '''second stage - accumulate in cells'''
        magnitude = np.sqrt(gx ** 2 + gy ** 2)
        orientation = np.arctan2(gy, gx) * (180 / np.pi) % 180

        '''my work - bin these just in the mask region'''
        these_magnitudes = magnitude.ravel()[mask.ravel()]
        these_orientations = orientation.ravel()[mask.ravel()]

        binned_orientations = (n_bins * (these_orientations / 180.0)).astype(np.int)

        new_hist = np.bincount(binned_orientations,
            weights=these_magnitudes, minlength=n_bins).astype(float)

        # note normalising by the number of points in mask.
        # the histogram won't generally sum to one - it instead captures
        # an overall degree of texture, so smoother regions have smaller
        # histogram sums and v.v.
        new_hist /= mask.sum()
        return new_hist

    def _get_rgb_hist(self, rgb):
        n_bins = 4
                # self.params['rgb_hist_bins']

        rgb_norm = rgb.astype(float) / 255
        bin_labels = np.floor(rgb_norm * (n_bins - 1)).astype(int)

        idxs = bin_labels[:, 0] * n_bins**2 \
             + bin_labels[:, 1] * n_bins \
             + bin_labels[:, 2]

        hist = np.bincount(idxs, minlength=n_bins**3).astype(float)
        assert hist.size == n_bins**3
        return hist / hist.sum()

    def _get_rgb_mean(self, rgb):
        mean = np.mean(rgb.astype(float)/255, 0)
        assert mean.size==3
        return mean

    def _get_bounding_box(self, xyz):
        # could change this to reject outliers
        print "NOTE: At the moment I am not doing the PCA like I should be"
        return np.max(xyz, 0) - np.min(xyz, 0)

    def _get_shape_dist(self, xyz):

        num_points = 5000
        p0 = xyz[np.random.choice(xyz.shape[0], num_points)]
        p1 = xyz[np.random.choice(xyz.shape[0], num_points)]
        dists = np.linalg.norm(p0 - p1, axis=1)

        # now form a histogram over these distances
        # slight modification in these edges from the matlab work
        edges = np.hstack([np.arange(0, 0.1, 0.01),
                           np.arange(0.1, 1.5, 0.1)])

        hist, _ = np.histogram(dists, edges)
        return hist.astype(float) / float(num_points)


def combine_features(feature_dict, features_to_use='all'):
    '''
    helper function to combine a dictioanry of features into a numpy array
    '''

    if features_to_use=='all':
        # must be careful to always retrieve items from dict in same order
        # so will sort the keys alphabetically
        keys = sorted([key for key in feature_dict])
    else:
        # will use the sorting as specified
        keys = features_to_use

    return np.hstack([feature_dict[key] for key in keys])