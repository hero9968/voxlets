'''
baseline algorithms
'''

import sys, os  # maxint
sys.path.append(os.path.expanduser("~/projects/shape_sharing/src/"))
from common import paths
from common import voxel_data

import numpy as np
import cv2
import copy

class BbBaseline(object):
    '''
    computes a best case bounding box
    axis aligned or something

    TODO - allow to set the upwards direction
    '''

    def __init__(self):
        pass

    def set_gt_vgrid(self, grid):
        self.vgrid = grid

    def compute_min_bb(self):

        # todo - modify the following if not using z as up dir
        full = np.any(self.vgrid.V>0.5, axis=2)
        hull_points_2d = np.array(np.nonzero(full)).T

        "Getting rectangle"
        print hull_points_2d
        rect = cv2.minAreaRect(hull_points_2d)
        box = cv2.cv.BoxPoints(rect)
        box = np.int0(box)

        "find z min and max"
        temp = np.any(self.vgrid.V>0.5, axis=0)
        temp = np.any(temp, axis=0)
        hull_points_z = np.nonzero(temp)
        min_z = np.min(hull_points_z)
        max_z = np.max(hull_points_z)

        single_layer = np.zeros((self.vgrid.V.shape[1], self.vgrid.V.shape[0]), np.uint8)
        cv2.fillConvexPoly(single_layer, box, 255, 0)

        # create output grid
        out_grid = copy.deepcopy(self.vgrid)
        out_grid.V *=0

        # fill each layer in the z direction
        for idx in range(min_z, max_z):
            out_grid.V[:, :, idx] = single_layer.T / 255.0

        out_grid.V = out_grid.V.astype(float)
        out_grid.convert_to_tsdf(0.03)
        return out_grid



class SimpleBbBaseline(object):
    '''
    computes a best case bounding box
    axis aligned or something

    TODO - allow to set the upwards direction
    '''

    def __init__(self):
        pass


    def set_gt_vgrid(self, grid):
        self.vgrid = grid


    def set_image(self, im):
        self.im = im



    def initialise_output_grid(self):

        # pad the gt grid slightly
        grid_origin = gt_grid.origin - 0.05
        grid_end = gt_grid.origin + np.array(gt_grid.V.shape).astype(float) * gt_grid.vox_size + 0.05

        voxlet_size = paths.voxlet_size/2.0
        grid_dims_in_real_world = grid_end - grid_origin
        V_shape = (grid_dims_in_real_world / (voxlet_size)).astype(int)
        print "Output grid will have shape " + str(V_shape)

        self.accum = voxel_data.UprightAccumulator(V_shape)
        self.accum.set_origin(grid_origin)
        self.accum.set_voxel_size(voxlet_size)


    def compute_min_bb(self):

        # get the world locations of the points in the image
        im_xyz = self.im.get_world_xyz()

        # extract only those in the mask
        im_xyz_in_mask = im_xyz[self.im.mask.flatten()==1, :]
        all_idx = self.vgrid.world_to_idx(im_xyz_in_mask)
        self.all_idx = all_idx

        # compute a minimum bb of these in the xy direction
        "Getting rectangle"
        print all_idx[:, :2].astype(np.float32)
        rect = cv2.minAreaRect(all_idx[:, :2].astype(np.float32))
        box = cv2.cv.BoxPoints(rect)
        box = np.int0(box)  

        print "Box is "
        print box       

        "find z max"
        print np.max(all_idx[:, 2])
        max_z = np.max(all_idx[:, 2])

        "find z min"
        # to be fair we will let the bounding box go down to the table surface
        temp = np.any(self.vgrid.V>0.5, axis=0)
        temp = np.any(temp, axis=0)
        hull_points_z = np.nonzero(temp)
        min_z = np.min(hull_points_z)  
        print min_z, max_z

        single_layer = np.zeros((self.vgrid.V.shape[1], self.vgrid.V.shape[0]), np.uint8)
        #self.single_layer = single_layer

        cv2.fillConvexPoly(single_layer, box, 255, 0)

        # create output grid
        out_grid = copy.deepcopy(self.vgrid)
        out_grid.V *=0

        # fill each layer in the z direction
        for idx in range(min_z, max_z):
            out_grid.V[:, :, idx] = single_layer.T / 255.0

        out_grid.V = out_grid.V.astype(float)
        #out_grid.convert_to_tsdf(0.03)
        return out_grid

