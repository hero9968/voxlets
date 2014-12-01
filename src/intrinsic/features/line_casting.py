import numpy as np
import line_casting_cython
import itertools

def line_features_2d(input_im):
    '''
    given an input image, computes the line features for each direction 
    and concatenates them somehow
    perhaps give options for how the features get returned, e.g. as 
    (H*W)*N or as a list...
    '''

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
    all_features = [line_casting_cython.outer_loop(input_im, direction) for direction in directions]
    return all_features








