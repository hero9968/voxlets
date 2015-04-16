import numpy as np
import socket

# some parameters can live here for now...
mu = 0.015
voxel_size = 0.005


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

pca_number_points_from_each_image = 100


class Voxlet(object):
    '''
    class for voxlet parameters
    '''
    # setting some voxlet params here
    # NOTE BE VERY CAREFUL IF EDITING THESE
    tall_voxlets = False

    one_side_bins = 20
    shape = (one_side_bins, 2*one_side_bins, one_side_bins)
    size = 0.0175 / 3.0  # edge size of a single voxel
    # centre is relative to the ijk origin at the bottom corner of the voxlet
    # z height of centre takes into account the origin offset
    actual_size = np.array(shape) * size
    centre = np.array((actual_size[0] * 0.5,
                       actual_size[1] * 0.25,
                       actual_size[2] * 0.5))

    tall_voxlet_height = np.nan
    # 0.175



class VoxletTraining(object):
    '''
    parameters for the training stage of the voxlet algorithm
    (Although the forest paramters are elsewhere currently)
    '''
    # PCA and kmeans
    pca_number_points_from_each_image = 500
    number_pca_dims = 60
    number_clusters = 50
    pca_subsample_length = 50000  # max number of examples to use for pca

    # actual voxlet extraction
    if small_sample:
        number_points_from_each_image = 250
        forest_subsample_length = 250000  # max num examples to use to train forest
    else:
        number_points_from_each_image = 1000
        forest_subsample_length = 500000  # max num examples to use to train forest

    decimation_rate = 2
    feature_transform = 'pca' # - what to do with the feature after extraction...



class VoxletPrediction(object):
    '''
    parameters for prediction stage of voxlet algorithm
    '''
    if small_sample:
        number_samples = 100  # number of points to sample from image
    else:
        number_samples = 100

    sampling_grid_size = 0.1
