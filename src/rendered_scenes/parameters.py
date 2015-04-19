import numpy as np
import socket


use_binary = False
scene_bagging = True

# general parameters about the system and the implementation
host_name = socket.gethostname()

if host_name == 'troll' or host_name == 'biryani':
    small_sample = False
    max_sequences = 500  # float('Inf')  # when in a loop, max num of images to use
    cores = 8
    multicore = True
else:
    small_sample = True
    max_sequences = 20
    cores = 4
    multicore = False


if small_sample:
    print "WARNING: Just computing on a small sample"


class RenderData(object):
    '''
    General parameters, e.g. about the overall experiemnts etc
    '''
    if small_sample:
        scenes_to_render = 20
        train_test_max_scenes = 30
        sequences_per_scene = 5
    else:
        scenes_to_render = 100
        train_test_max_scenes = 100
        sequences_per_scene = 10

    # train test split parameters
    frames_per_sequence = 1
    train_fraction = 0.6


class Voxlet(object):
    '''
    class for voxlet parameters
    '''
    # setting some voxlet params here
    # NOTE BE VERY CAREFUL IF EDITING THESE
    tall_voxlets = True

    one_side_bins = 20
    shape = (one_side_bins, 2*one_side_bins, 2*one_side_bins)
    size = 0.0175  # edge size of a single voxel
    # centre is relative to the ijk origin at the bottom corner of the voxlet
    # z height of centre takes into account the origin offset
    actual_size = np.array(shape) * size
    centre = np.array((actual_size[0] * 0.5,
                       actual_size[1] * 0.25,
                       0.375+0.03))

    tall_voxlet_height = 0.375


class RenderedVoxelGrid(object):
    '''
    parameters for the voxel grid established for the rendered data
    '''
    shape = np.array((150, 150, 75))  # number of voxels in each direction
    voxel_size = 0.01  # edge length of a voxel
    origin = np.array((-0.75, -0.75, -0.03))
    R = np.eye(3)

    mu = 0.1  # truncation parameter for the tsdf

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
    use_implicit = False  # should we use the implicit predictions as a feature?

    # PCA and kmeans
    pca_number_points_from_each_image = 50
    number_pca_dims = 400
    number_clusters = 250
    pca_subsample_length = 50000  # max number of examples to use for pca

    # actual voxlet extraction
    if small_sample:
        number_points_from_each_image = 250
        forest_subsample_length = 25000  # max num examples to use to train forest
    else:
        number_points_from_each_image = 500
        forest_subsample_length = 50000  # max num examples to use to train forest

    decimation_rate = 2
    feature_transform = 'pca' # - what to do with the feature after extraction...




class VoxletPrediction(object):
    '''
    parameters for prediction stage of voxlet algorithm
    '''
    if small_sample:
        number_samples = 300  # number of points to sample from image
    else:
        number_samples = 300

    sampling_grid_size = 0.1
