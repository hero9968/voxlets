multicore = False
cores = 4

import numpy as np


class Voxlet(object):
    '''
    defining a class for voxlet parameters for these scenes, so we can adjust them...
    '''
    # setting some voxlet params here
    # NOTE BE VERY CAREFUL IF EDITING THESE
    tall_voxlets = True

    one_side_bins = 20
    shape = (one_side_bins, 2*one_side_bins, 2*one_side_bins)
    size = 0.0175 / 1.3  # edge size of a single voxel
    # centre is relative to the ijk origin at the bottom corner of the voxlet
    # z height of centre takes into account the origin offset
    actual_size = np.array(shape) * size

    tall_voxlet_height = actual_size[2] /2

    centre = np.array((actual_size[0] * 0.5,
                       actual_size[1] * 0.25,
                       tall_voxlet_height-0.03))

class RenderedVoxelGrid(object):
    mu = 0.1

class VoxletPrediction(object):
    number_samples = 200
    sampling_grid_size = 0.1