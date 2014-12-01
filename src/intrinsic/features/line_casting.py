import numpy as np
import line_casting_cython
import itertools
import copy

out_of_range_value = 250

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
    itertools_directions = itertools.product([-1, 0, 1], repeat=2)
    directions = np.array(list(itertools_directions)).astype(np.int32)

    # removing the [0, 0, 0] entry
    to_remove = np.sum(np.abs(directions), 1) == 0
    directions = directions[~to_remove]

    # sorting array by the angle (polar coordinates representation)
    idxs = np.argsort((np.arctan2(directions[:, 1], directions[:, 0])))
    directions = directions[idxs]

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




