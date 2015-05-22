import yaml
import math, random
import numpy as np
from sklearn.neighbors import NearestNeighbors

import sys, os

sys.path.append(os.path.expanduser('~/projects/shape_sharing/src/'))
sys.path.append(os.path.expanduser('~/projects/shape_sharing/src/intrinsic/'))

from common import scene, voxel_data
from features import line_casting


def fibonacci_sphere(samples=1,randomize=True):
    '''
    From the internet
    '''
    rnd = 1.
    if randomize:
        rnd = random.random() * samples

    points = []
    offset = 2./samples
    increment = math.pi * (3. - math.sqrt(5.));

    for i in range(samples):
        y = ((i * offset) - 1) + (offset / 2);
        r = math.sqrt(1 - pow(y,2))

        phi = ((i + rnd) % samples) * increment

        x = math.cos(phi) * r
        z = math.sin(phi) * r

        points.append([x,y,z])

    return points


def norm_v(vec):
    return vec / np.linalg.norm(vec)


def find_axes(norms_in):
    '''
    Implementation of section 2.2 of
    http://grail.cs.washington.edu/projects/manhattan/manhattan.pdf
    '''
    assert norms_in.shape[1] == 3

    to_use = ~np.any(np.isnan(norms_in), axis=1)
    norms = norms_in[to_use, :]

    # forming a histogram over the normals
    # (using twice as many bins as them, but normals will only point towards camera
    # so there will in fact only be 1000 used)
    n_bins = 2000
    bins = np.array(fibonacci_sphere(n_bins))

    nbrs = NearestNeighbors(n_neighbors=1).fit(bins)
    _, idxs = nbrs.kneighbors(norms)

    idxs = idxs.flatten()
    histogram = np.bincount(idxs, minlength=n_bins)

    # finding the peak in this histogram and the avg direction of all normals in this bin
    peak_bin_idx = np.argmax(histogram)
    inliers = norms[idxs==peak_bin_idx]
    d1 = norm_v(np.mean(inliers, axis=0))

    # now finding bins which are within 80-100 degrees of this peak direction
    # i.e. within 10 degrees of 90
    abs_upper_bound = np.arcsin(np.deg2rad(10))
    bins_approx_perpendicular = np.abs(np.dot(d1, bins.T)) < abs_upper_bound

    histogram[~bins_approx_perpendicular] = 0


    if histogram.max() == 0:
        # no good bins, so just take a random direction here
        d2_fuzzy = np.random.rand(3)
    else:
        peak_bin_idx = np.argmax(histogram)
        inliers = norms[idxs==peak_bin_idx]
        d2_fuzzy = norm_v(np.mean(inliers, axis=0))

    # now I will deviate slightly from the paper, and here I will actually
    # force the directions to be perpendicular
    d3 = norm_v(np.cross(d1, d2_fuzzy))
    d2 = norm_v(np.cross(d3, d1))

    R = np.vstack((d1, d2, d3))

    return R


def process_scene(sc, threshold=3):

    segments = np.unique(sc.gt_im_label)[1:]
    accumulator = sc.gt_tsdf.blank_copy()
    print accumulator.V.dtype

    for seg in segments:
        print "Doing segment ", seg

        inlying_voxels = sc.gt_labels == seg
        inlying_pixels = sc.gt_im_label.flatten() == seg

        # find the principal directinos of the segment
        norms = sc.im.get_world_normals()[inlying_pixels]
        R = find_axes(norms)

        # first rotate all the voxels
        world_voxels = sc.gt_tsdf.world_meshgrid()
        rotated_voxels = np.dot(R, world_voxels.T).T

        # now find the extents of this rotated grid...
        min_grid = rotated_voxels.min(axis=0)
        max_grid = rotated_voxels.max(axis=0)

        # use these extents to find the posdt-rotation offset required
        offset = min_grid
        grid_size = np.ceil((max_grid - min_grid) / sc.gt_tsdf.vox_size)

        # now rotate these labels to the new space
        known_full = sc.gt_tsdf.blank_copy()
        new_origin = np.dot(np.linalg.inv(R), offset)
        known_full.set_origin(new_origin, np.linalg.inv(R))
        known_full.V = np.zeros(grid_size)

        # but I want to find the edges which correspond to this segment
        region_edges = sc.im_visible.copy()
        region_edges.V[sc.gt_labels.V!=seg] = 0
        known_full.fill_from_grid(region_edges)

        # also need to rotate a copy of the empty grid
        empty_voxels = sc.im_tsdf.copy()
        empty_voxels.V = empty_voxels.V > 0
        known_empty = known_full.blank_copy()
        known_empty.fill_from_grid(empty_voxels)

        # (if desired I could crop this grid down...)

        # now I want to use my routine to fill in the grid
        distances, observed = line_casting.line_features_3d(known_empty, known_full, base_height=0)

        # now running the zheng thresholding
        # only using certain directions from all the directions I have computed!
        to_use = [0, 9, 11, 13, 15, 25]
        all_observed = np.vstack([observed[ii].flatten() for ii in to_use]).T
        predicted_full = all_observed.sum(1) >= threshold

        # now rotate these labels to the new space
        this_pred_grid = known_full.blank_copy()
        this_pred_grid.V = predicted_full.reshape(observed[0].shape)

        rotated_back = sc.gt_tsdf.blank_copy()
        rotated_back.fill_from_grid(this_pred_grid)

        accumulator.V = np.logical_or(rotated_back.V > 0, accumulator.V > 0)

    return accumulator