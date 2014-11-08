'''
computes a best case bounding box
axis aligned or something
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
            out_grid.V[:, :, idx] = single_layer.T

        return out_grid

