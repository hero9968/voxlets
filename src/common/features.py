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
# from sklearn.neighbors import KDTree

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib as mpl


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

    def __init__(self, t, fixed_patch_size=False):

        # the stepsize at a depth of 1 m
        self.t = float(t)

        # dimension of side of patch in real world 3D coordinates
        #self.input_patch_hww = input_patch_hww

        # if fixed_patch_size is True:
        #   step is always t in input image pixels
        # else:
        #   step varies linearly with depth. t is the size of step at depth of 1.0
        self.fixed_patch_size = fixed_patch_size

    def set_image(self, im):
        self.im = im

    def get_cobweb(self, index):
        '''extracts cobweb for a single index point'''
        row, col = index

        start_angle = 0#self.im.angles[row, col]
        start_depth = self.im.depth[row, col]

        focal_length = self.im.cam.estimate_focal_length()
        if self.fixed_patch_size:
            offset_dist = focal_length * self.t
        else:
            offset_dist = (focal_length * self.t) / start_depth

        # computing all the offsets and angles efficiently
        offsets = offset_dist * np.array([1, 2, 3, 4, 5, 6])
        rad_angles = np.deg2rad(start_angle + np.array(range(0, 360, 45)))

        rows_to_take = (float(row) - np.outer(offsets, np.sin(rad_angles))).astype(int).flatten()
        cols_to_take = (float(col) + np.outer(offsets, np.cos(rad_angles))).astype(int).flatten()

        # defining the cobweb array ahead of time
        fv_length = rows_to_take.shape[0]  # == cols_to_take.shape[0]
        cobweb = np.nan * np.zeros((fv_length, )).flatten()

        # working out which indices are within the image bounds
        to_use = np.logical_and.reduce((rows_to_take >= 0,
                                        rows_to_take < self.im.depth.shape[0],
                                        cols_to_take >= 0,
                                        cols_to_take < self.im.depth.shape[1]))
        rows_to_take = rows_to_take[to_use]
        cols_to_take = cols_to_take[to_use]

        # computing the diff vals and slotting into the correct place in the cobweb feature
        vals = self.im.depth[rows_to_take, cols_to_take] - self.im.depth[row, col]
        cobweb[to_use] = vals
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

    def compute_normals(self, im):
        '''
        one method of computing normals
        '''
        xyz = im.reproject_3d()

        x = xyz[0, :].reshape(im.depth.shape)
        y = xyz[1, :].reshape(im.depth.shape)
        z = xyz[2, :].reshape(im.depth.shape)

        dx0, dx1 = np.gradient(x)
        dy0, dy1 = np.gradient(y)
        dz0, dz1 = np.gradient(z)

        dxyz0 = np.vstack((dx0.flatten(), dy0.flatten(), dz0.flatten()))
        dxyz1 = np.vstack((dx1.flatten(), dy1.flatten(), dz1.flatten()))
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

    # def kdtree_normals(self, im):
    #     '''
    #     '''
    #     xyz = im.reproject_3d().T
    #     nans = np.any(np.isnan(xyz), axis=1)
    #     print nans.shape, nans.sum()
    #     print "xyz is shape", xyz.shape
    #     tree = KDTree(xyz[~nans, :])
    #     _, xyz_neighbours = tree.query(xyz[~nans, :], k=20)
    #     print xyz_neighbours.shape


    def voxel_normals(self, im, vgrid):
        '''
        compute the normals from a voxel grid
        '''
        offset = 4
        xyz = im.get_world_xyz()
        inliers = np.ravel(im.mask)

        # padding the array
        pad_width = ((offset, offset), (offset, offset), (offset, offset))
        padded = np.pad(vgrid.V, pad_width, 'edge')
        padded[np.isnan(padded)] = np.nanmin(padded)

        idx = vgrid.world_to_idx(xyz[inliers]) + offset

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
        print np.isnan(diffs).sum()
        print diffs.shape

        # now convert the normals to image space instead of world space...
        image_norms = np.dot(im.cam.inv_H[:3, :3], diffs).T

        # pad out array to the correct size for future computations...
        output_norms = np.zeros((im.mask.size, 3), dtype=np.float32)
        output_norms[inliers, :] = image_norms
        return output_norms


        # return diffs.T
