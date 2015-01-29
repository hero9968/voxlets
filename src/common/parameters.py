import numpy as np
import socket


# general parameters about the system and the implementation
host_name = socket.gethostname()

if host_name == 'troll' or host_name == 'biryani':
    small_sample = False
    cores = 8
    multicore = True
else:
    small_sample = True
    cores = 4
    multicore = False


if small_sample:
    print "WARNING: Just computing on a small sample"


class General(object):
    '''
    General parameters, e.g. about the overall experiemnts etc
    '''
    if small_sample:
        scenes_to_render = 4
    else:
        scenes_to_render = 100


class Voxlet(object):
    '''
    class for voxlet parameters
    '''
    # setting some voxlet params here
    # NOTE BE VERY CAREFUL IF EDITING THESE
    one_side_bins = 15
    shape = (one_side_bins, 2*one_side_bins, one_side_bins)
    size = 0.2/float(one_side_bins)
    centre = np.array((0.1, 0.05, 0.1))


class RenderedVoxelGrid(object):
    '''
    parameters for the voxel grid established for the rendered data
    '''
    shape = np.array((150, 150, 75))  # number of voxels in each direction
    voxel_size = 0.01  # edge length of a voxel
    origin = np.array((-0.75, -0.75, -0.03))
    R = np.eye(3)

    mu = 0.05  # truncation parameter for the tsdf

    @classmethod
    def aabb(cls):
        '''
        returns the axis aligned bounding box as a pair of tuples
        '''
        grid_min = cls.origin
        grid_max = cls.origin + cls.voxel_size * cls.shape
        return grid_min, grid_max


class VoxletTraining(object):
    '''
    parameters for the training stage of the voxlet algorithm
    (Although the forest paramters are elsewhere currently)
    '''
    cobweb_t = 5  # this is the parameter of the cobweb feature extraction...

    # PCA and kmeans
    pca_number_points_from_each_image = 50
    number_pca_dims = 50
    number_clusters = 150
    pca_subsample_length = 25000  # max number of examples to use for pca

    # actual voxlet extraction
    number_points_from_each_image = 100

    forest_subsample_length = 25000  # max num examples to use to train forest


class VoxletPrediction(object):
    '''
    parameters for prediction stage of voxlet algorithm
    '''
    number_samples = 200  # number of points to sample from image