import numpy as np
import line_casting_cython
import itertools
import copy

out_of_range_value = 100


def get_directions_2d():
    '''
    doing this by hand instead of using e.g. itertools, for clarity.
    also to preserve angle order and as we don't want [0, 0] to be included
    '''
    return [[0, -1],
           [1, -1],
           [1, 0],
           [1, 1],
           [0, 1],
           [-1, 1],
           [-1, 0],
           [-1, -1]]


def get_directions_3d():
    '''
    speed is not an issue here, but clarity and bug-free-ness is
    Writing this 'by hand' could introduce bugs, so instead make 
    use of the 2d version with a loop for each z-direction
    '''
    directions_2d = get_directions_2d()

    dir1 = [direction + [-1] for direction in directions_2d]
    dir2 = [direction + [0] for direction in directions_2d]
    dir3 = [direction + [1] for direction in directions_2d]

    return [[0, 0, -1]] + dir1 + dir2 + dir3 + [[0, 0, 1]]


def line_features_2d(observed_tsdf, known_filled):
    '''
    given an input image, computes the line features for each direction 
    and concatenates them somehow
    perhaps give options for how the features get returned, e.g. as 
    (H*W)*N or as a list...
    '''

    # constructing the input image from the two inputs
    input_im = copy.deepcopy(observed_tsdf) * 0 - 1
    input_im[observed_tsdf > 0] = 0
    input_im[known_filled==1] = 1

    # generating numpy array of directions
    directions = np.array(get_directions_2d()).astype(np.int32)

    # computing the actual features using the cython code
    all_distances = []
    all_observed = []
    for direction in directions:
        distances, observed = line_casting_cython.outer_loop(input_im, direction)
        distances[distances==-1] = out_of_range_value
        all_distances.append(distances)
        all_observed.append(observed)

    # returning the data as specified
    return all_distances, all_observed


def feature_pairs(observed_tsdf, known_filled,  gt_tsdf, samples=-1):
    '''
    samples 
        is an integer defining how many feature pairs to sample. 
        If samples==-1, all feature pairs are returned
    '''

    all_distances, all_observed = line_features_2d(observed_tsdf, known_filled)

    # converting computed features to reshaped numpy arrays
    N = len(all_distances)
    all_distances_np = np.array(all_distances).astype(np.int16).reshape((N, -1)).T
    all_observed_np = np.array(all_observed).astype(np.int16).reshape((N, -1)).T

    # get feature pairs from the cast lines
    unknown_voxel_idxs = observed_tsdf.flatten() < 0
    Y = gt_tsdf.flatten()[unknown_voxel_idxs]

    X1 = all_distances_np[unknown_voxel_idxs]
    X2 = all_observed_np[unknown_voxel_idxs]
    X = np.concatenate((X1, X2), axis=1)

    # subsample if requested
    if samples > -1:
        idx_to_use = np.random.choice(Y.shape[0], samples, replace=False)
        Y = Y[idx_to_use]
        X = X[idx_to_use]

    return X, Y


def line_features_3d(known_empty_voxels, known_full_voxels):
    #Not done yet!
    '''
    given an input image, computes the line features for each direction 
    and concatenates them somehow
    perhaps give options for how the features get returned, e.g. as 
    (H*W)*N or as a list...
    '''
    input_im = known_full_voxels.blank_copy()
    input_im.V += -1
    input_im.V[known_full_voxels.V==1] = 1
    input_im.V[known_empty_voxels.V==1] = 0

    # generating numpy array of directions
    # note that in my coordinate system, up is k
    directions = np.array(get_directions_3d()).astype(np.int32)

    # computing the actual features using the cython code
    all_distances = []
    all_observed = []
    for direction in directions:
        distances, observed = line_casting_cython.outer_loop_3d(input_im.V.astype(np.int8), direction)
        
        # scaling distances so they are equal in real-world terms
        distances = (distances.astype(np.float) *  np.linalg.norm(direction)).astype(np.int32)

        # dealing with out of range distances
        distances[distances==-1] = out_of_range_value
        
        all_distances.append(distances)
        all_observed.append(observed)

    # returning the data as specified
    return all_distances, all_observed


def feature_pairs_3d(known_empty_voxels, known_full_voxels, gt_tsdf, samples=-1, base_height=0):
    '''
    samples 
        is an integer defining how many feature pairs to sample. 
        If samples==-1, all feature pairs are returned

    base_height
        is an integer defining how many voxels to ignore from the base of the grid
    '''

    all_distances, all_observed = line_features_3d(known_empty_voxels, known_full_voxels)

    # converting computed features to reshaped numpy arrays
    N = len(all_distances)
    all_distances_np = np.array(all_distances).astype(np.int16).reshape((N, -1)).T
    all_observed_np = np.array(all_observed).astype(np.int16).reshape((N, -1)).T

    # get feature pairs from the cast lines
    unknown_voxel_idxs = observed_tsdf.flatten() < 0
    Y = gt_tsdf.flatten()[unknown_voxel_idxs]

    X1 = all_distances_np[unknown_voxel_idxs]
    X2 = all_observed_np[unknown_voxel_idxs]
    X = np.concatenate((X1, X2), axis=1)

    # subsample if requested
    if samples > -1:
        idx_to_use = np.random.choice(Y.shape[0], samples, replace=False)
        Y = Y[idx_to_use]
        X = X[idx_to_use]

    return X, Y