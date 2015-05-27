import numpy as np
import line_casting_cython
import itertools
import copy
import scipy.ndimage
out_of_range_value = 500


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


def roll_rows_independently(A, r):
    '''
    like np.roll, but allows each row to be rolled indepentdently
    http://stackoverflow.com/questions/20360675/roll-rows-of-a-matrix-independently
    '''
    rows, column_indices = np.ogrid[:A.shape[0], :A.shape[1]]

    # Use always a negative shift, so that column_indices are valid.
    # (could also use module operation)
    r %= A.shape[1]
    column_indices = column_indices - r[:,np.newaxis]

    return  A[rows, column_indices]


def autorotate_3d_features(distances, observed):
    '''
    circshifts the feature vector about the z-axis such that the
    direction in the horizontal plane which hits an observed voxel
    first is the first feature given.
    hardcodes the ordering of the features so be careful!
    feature parts are: a[0], a[1:9], a[9:17], a[17:25], a[25]
    '''
    assert(distances.shape[1] == 26)
    assert(observed.shape[1] == 26)

    # extracting the distances which are horizontal
    observed_subpart = observed[:, 9:17]
    distances_subpart = distances[:, 9:17].copy()

    # setting those distances which don't hit oberserved geometry to be very large
    distances_subpart[observed_subpart!=1] = 1e6

    # find the minimum of all the distances now
    min_direction_idx = np.argmin(distances_subpart, axis=1)

    # now rotate all the points
    observed[:, 1:9] = roll_rows_independently(observed[:, 1:9], -min_direction_idx)
    observed[:, 9:17] = roll_rows_independently(observed[:, 9:17], -min_direction_idx)
    observed[:, 17:25] = roll_rows_independently(observed[:, 17:25], -min_direction_idx)

    distances[:, 1:9] = roll_rows_independently(distances[:, 1:9], -min_direction_idx)
    distances[:, 9:17] = roll_rows_independently(distances[:, 9:17], -min_direction_idx)
    distances[:, 17:25] = roll_rows_independently(distances[:, 17:25], -min_direction_idx)

    return distances, observed

def sort_feature_set(distances, observed):
    '''
    sorts just one set of 8 directions
    '''
    assert(distances.shape[1] == 8)
    assert(observed.shape[1] == 8)

    idxs = np.argsort(distances, axis=1) + np.arange(observed.shape[0])[:, None] * 8
    shape = distances.shape

    return (distances.flatten()[idxs].reshape(shape),
            observed.flatten()[idxs].reshape(shape))


def sort_3d_features(distances, observed):
    '''
    sorts each set of 8 directions separately
    '''
    assert(distances.shape[1] == 26)
    assert(observed.shape[1] == 26)

    distances[:, 1:9], observed[:, 1:9] = \
        sort_feature_set(distances[:, 1:9], observed[:, 1:9])

    distances[:, 9:17], observed[:, 9:17] = \
        sort_feature_set(distances[:, 9:17], observed[:, 9:17])

    distances[:, 17:25], observed[:, 17:25] = \
        sort_feature_set(distances[:, 17:25], observed[:, 17:25])

    return distances, observed


def sort_3d_features_together(distances, observed):
    '''
    sorts all the direction sets together to match the middle one
    '''
    assert(distances.shape[1] == 26)
    assert(observed.shape[1] == 26)

    idxs = np.argsort(distances[:, 9:17], axis=1) + \
        np.arange(observed.shape[0])[:, None] * 8

    shape = distances[:, 9:17].shape

    for start in [1, 9, 17]:
        distances[:, start:(start+8)] = \
            distances[:, start:(start+8)].flatten()[idxs].reshape(shape)
        observed[:, start:(start+8)] = \
            observed[:, start:(start+8)].flatten()[idxs].reshape(shape)

    return distances, observed

import scipy.io

def line_features_3d(known_empty_voxels, known_full_voxels, base_height=0):
    #Not done yet!
    '''
    given an input image, computes the line features for each direction
    and concatenates them somehow
    perhaps give options for how the features get returned, e.g. as
    (H*W)*N or as a list...
    autorotate decides if the feature vectors are circshifted so the closest point is at the start...
    '''
    # remove the base voxels
    known_empty_voxels.V = known_empty_voxels.V[:, :, base_height:]
    known_full_voxels.V = known_full_voxels.V[:, :, base_height:]

    input_im = known_full_voxels.blank_copy()
    input_im.V = input_im.V.astype(np.int8)
    input_im.V += -1
    input_im.V[known_empty_voxels.V==1] = 0

    # dilate the known full voxels by one
    dilated_known_full_voxels_V = \
        scipy.ndimage.binary_dilation(known_full_voxels.V).astype(known_full_voxels.V.dtype)
    input_im.V[dilated_known_full_voxels_V==1] = 1

    #scipy.io.savemat('/tmp/input_im.mat', dict(input_im=input_im.V))

    # generating numpy array of directions
    # note that in my coordinate system, up is k
    directions = np.array(get_directions_3d()).astype(np.int32)

    # computing the actual features using the cython code
    all_distances = []
    all_observed = []
    for count, direction in enumerate(directions):
        distances, observed = line_casting_cython.outer_loop_3d(input_im.V.astype(np.int8), direction)

        # scaling distances so they are equal in real-world terms (note np.linalg.norm returns a float)
        distances = (distances.astype(np.float) *  np.linalg.norm(direction)).astype(np.int32)

        # dealing with out of range distances
        distances[distances==-1] = out_of_range_value
        # scipy.io.savemat('/tmp/distances%03d.mat' % count, dict(distances=distances))


        all_distances.append(distances)
        all_observed.append(observed)

    # returning the data as specified
    return all_distances, all_observed


def feature_pairs_3d(
    known_empty_voxels,
    known_full_voxels,
    gt_tsdf=None,
    samples=-1,
    base_height=0,
    postprocess=None,
    all_voxels=False,
    in_frustrum=None):
    '''
    samples
        is an integer defining how many feature pairs to sample.
        If samples==-1, all feature pairs are returned

    base_height
        is an integer defining how many voxels to ignore from the base of the grid
    '''
    base_path = '/Users/Michael/projects/shape_sharing/data/'\
        'rendered_arrangements/test_sequences/dm779sgmpnihle9x/'
    all_distances, all_observed = line_features_3d(known_empty_voxels, known_full_voxels)
    #scipy.io.savemat(base_path + 'all_distances.mat', dict(all_distances=all_distances))
    # converting computed features to reshaped numpy arrays
    N = len(all_distances)
    all_distances_np = np.array(all_distances).astype(np.int16).reshape((N, -1)).T
    all_observed_np = np.array(all_observed).astype(np.int16).reshape((N, -1)).T

    if postprocess == 'autorotate':
        print "autorotating"
        all_distances_np, all_observed_np = \
            autorotate_3d_features(all_distances_np, all_observed_np)
    elif postprocess == 'sort':
        print "sorting"
        all_distances_np, all_observed_np = \
            sort_3d_features(all_distances_np, all_observed_np)
    elif postprocess == 'sort_together':
        print "sorting together"
        all_distances_np, all_observed_np = \
            sort_3d_features_together(all_distances_np, all_observed_np)
    elif postprocess:
        raise Exception('Unknown postprocess method %s' % postprocess)

    # get feature pairs from the cast lines
    if all_voxels:
        voxels_to_use = np.ones(known_empty_voxels.flatten().shape, dtype=bool)
    else:
        # just use the voxels which are both in the camera frustrum and
        # behind the depth image
        assert(in_frustrum is not None)

        voxels_to_use = np.logical_and.reduce((
            known_empty_voxels.V.flatten() == 0,
            known_full_voxels.V.flatten() == 0,
            in_frustrum))

    Y = gt_tsdf.flatten()[voxels_to_use]
    X1 = all_distances_np[voxels_to_use]
    X2 = all_observed_np[voxels_to_use]

    X = np.concatenate((X1, X2), axis=1)

    # subsample if requested
    if samples > -1:
        idx_to_use = np.random.choice(Y.shape[0], samples, replace=False)
        Y = Y[idx_to_use]
        X = X[idx_to_use]
        voxels_to_use = np.where(voxels_to_use)[0][idx_to_use]

    return X, Y, voxels_to_use


def unique_rows(data):
    ncols = data.shape[1]
    dtype = data.dtype.descr * ncols
    struct = data.view(dtype)

    uniq, inv = np.unique(struct, return_inverse=True)
    return uniq.view(data.dtype).reshape(-1, ncols), inv

import sys, os
sys.path.append(os.path.expanduser('~/projects/shape_sharing/src/'))
from common import features

def cobweb_distance_features(sc, voxels_to_use, cobweb_offset):

    # computing the depth of each of these voxels
    xyz = sc.im_tsdf.world_meshgrid()[voxels_to_use]
    projected_voxels = sc.im.cam.project_points(xyz)

    uv = np.round(projected_voxels[:, :2]).astype(int)
    inside_image = np.logical_and.reduce((uv[:, 0] >= 0,
                                          uv[:, 1] >= 0,
                                          uv[:, 1] < sc.im.depth.shape[0],
                                          uv[:, 0] < sc.im.depth.shape[1]))

    uv = uv[inside_image]
    depths = projected_voxels[inside_image, 2]

    # computing the depth behind the input image of each of these voxels
    observed_depths = sc.im.depth[uv[:, 1], uv[:, 0]]
    surface_to_voxel_dist_s = depths - observed_depths

    # aim here is to only compute the cobweb feature for each point once,
    # but allow each voxel to then use that feature...

    # finding which are the pixels which the points project to
    idxs, inv = unique_rows(uv)

    # compute the cobweb features for every point on the depth image...
    ce = features.CobwebEngine(cobweb_offset)
    ce.set_image(sc.im)
    # cobweb engine wants row then column
    cobweb = np.array(ce.extract_patches(idxs[:, ::-1]))

    # now re-expand out to the full dimensions
    return np.hstack((cobweb[inv], surface_to_voxel_dist_s[:, None]))
