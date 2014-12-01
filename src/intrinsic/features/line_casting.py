import numpy as np
import line_casting_cython
import itertools
import copy

def line_features_2d(known_empty, known_filled):
    '''
    given an input image, computes the line features for each direction 
    and concatenates them somehow
    perhaps give options for how the features get returned, e.g. as 
    (H*W)*N or as a list...
    '''

    # constructing the input image from the two inputs
    input_im = copy.deepcopy(known_empty) * 0 - 1
    input_im[known_empty==1] = 0
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
        all_distances.append(distances)
        all_observed.append(observed)
    
    return all_distances, all_observed








