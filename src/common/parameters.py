import numpy as np
import socket


# general parameters about the system and the implementation
host_name = socket.gethostname()

if host_name == 'troll' or host_name == 'biryani':
    small_sample = False
    cores = 8
else:
    small_sample = True
    cores = 4


if small_sample:
    print "WARNING: Just computing on a small sample"


class Voxlet(object):
    '''
    class for voxlet parameters
    '''
    # setting some voxlet params here
    # NOTE BE VERY CAREFUL IF EDITING THESE
    one_side_bins = 15
    shape = (one_side_bins, 2*one_side_bins, one_side_bins)
    size = 0.1/float(one_side_bins)
    centre = np.array((0.05, 0.025, 0.05))


class RenderedVoxelGrid(object):
    '''
    parameters for the voxel grid established for the rendered data
    '''
    shape = np.array((150, 150, 75))  # number of voxels in each direction
    voxel_size = 0.01  # edge length of a voxel
    origin = np.array((-0.75, -0.75, -0.03))
    R = np.eye(3)

    @classmethod
    def aabb(cls):
        '''
        returns the axis aligned bounding box as a pair of tuples
        '''
        grid_min = cls.origin
        grid_max = cls.origin + cls.voxel_size * cls.shape
        return grid_min, grid_max

