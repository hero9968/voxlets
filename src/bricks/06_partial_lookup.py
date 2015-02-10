import sys
import os
import numpy as np
import cPickle as pickle
from sklearn.neighbors import NearestNeighbors
from itertools import starmap

import bricks
sys.path.append(os.path.expanduser('~/projects/shape_sharing/src/'))
from common import voxel_data
from common import images
from common import carving
from common import paths
from common import parameters

brick_side = brick_side = (10, 10, 75)

print "Loading mean and cov..."

real = True
random_nearest_neighbours = True
# with open('data/cov.pkl', 'rb') as f:
#     mean, cov = pickle.load(f)

with open('data/training_data.pkl', 'rb') as f:
    training_data = pickle.load(f)

print np.array(training_data).shape
print training_data.shape
print training_data.dtype

#@profile
def fill_in_missing(idx_data):
    '''
    fills in missing (masked) values in data
    '''
    idx, data = idx_data
    print idx
    mask = np.isnan(data)

    if mask.sum() == 0:
        return data
    elif mask.sum() == mask.shape[0]:
        print "Just mean"
        return np.mean(training_data, axis=0)
    else:

        if real and np.random.rand() > 0.95:
            # if random_nearest_neighbours:
            #     print "Before: ", mask.sum()
            #     mask[mask]
            #     print "After: ", mask.sum()

            lookup_table = training_data[:, ~mask]
            nbrs = NearestNeighbors(n_neighbors=1, algorithm='brute')
            nbrs.fit(lookup_table)
            distances, indices = nbrs.kneighbors(data[~mask])
            print "Real...", indices[0][0]
            return training_data[indices[0][0]]

        else:
            print "Cheating"
            return training_data[0]

import multiprocessing
import functools
pool = multiprocessing.Pool(parameters.cores)
print parameters.cores

for count, sequence in enumerate(paths.RenderedData.train_sequence()[:1]):

    print "Processing " + sequence['name']

    # load in the ground truth grid for this scene, and converting nans
    gt_vox = voxel_data.load_voxels(
        paths.RenderedData.ground_truth_voxels(sequence['scene']))
    gt_vox.V[np.isnan(gt_vox.V)] = -parameters.RenderedVoxelGrid.mu
    gt_vox.set_origin(gt_vox.origin)

    # this is where to
    input_data_path = paths.RenderedData.video_yaml(sequence['scene'])
    vid = images.RGBDVideo()
    vid.load_from_yaml(input_data_path)

    # now create the partial tsdf...
    carver = carving.Fusion()
    carver.set_video(vid.subvid(sequence['frames']))
    partial_tsdf = gt_vox.blank_copy()
    carver.set_voxel_grid(partial_tsdf)
    partial_tsdf, visible = carver.fuse(mu=parameters.RenderedVoxelGrid.mu)

    # split this partial tsdf into bricks using the bricks code
    brick_grid = bricks.Bricks()
    brick_grid.from_voxel_grid(partial_tsdf.V, brick_side)
    partial_bricks = brick_grid.to_flat()

    # try to complete each brick using the cov matrix
    #predicted_bricks = map(fill_in_missing, partial_bricks, range(len(partial_bricks)))
    predicted_bricks = pool.map(fill_in_missing, enumerate(partial_bricks))
    #predicted_bricks = map(fill_in_missing, enumerate(partial_bricks))
    predicted_bricks = np.array(predicted_bricks)

    # now recreate the original thing from the partial bricks
    print predicted_bricks.shape

    guessed_bricks = brick_grid.blank_copy()
    guessed_bricks.from_flat(predicted_bricks)
    guessed_V = guessed_bricks.to_voxel_grid()

    guessed_vox = gt_vox.blank_copy()
    guessed_vox.V = guessed_V

    # render output
    renderpath = paths.Bricks.prediction_frame % \
        ('zbrick_partial', sequence['name'], 'result')
    guessed_vox.render_view(renderpath)


    renderpath = paths.Bricks.prediction_frame % \
        ('zbrick_partial', sequence['name'], 'visible')
    visible.render_view(renderpath)


    # saving to disk



    # creating the training set.

    # for each brick, do a partial lookup into the training set to try to find
    # a good completion...

    break

    # loadpath = paths.RenderedData.ground_truth_voxels(scenename)
    # vox = voxel_data.load_voxels(loadpath)
    # vox.V[np.isnan(vox.V)] = -parameters.RenderedVoxelGrid.mu

    # brick_grid = bricks.Bricks()
    # brick_grid.from_voxel_grid(vox.V, brick_side)
    # this_scene_examples = brick_grid.to_flat()

    # pca_bricks = pca.transform(this_scene_examples)
    # if oracle_type == 'kmeans':
    #     cluster_idxs = km.predict(pca_bricks)
    #     closest_clusters_pca = km.cluster_centers_[cluster_idxs]
    #     closest_clusters = pca.inverse_transform(closest_clusters_pca)
    # elif oracle_type == 'pca':
    #     closest_clusters = pca.inverse_transform(pca_bricks)

    # brick_grid.from_flat(closest_clusters)
    #     # closest_clusters,
    #     # brick_grid.shape[:3],
    #     # brick_side=brick_side,
    #     # original_shape=vox.V.shape)

    # #print reformed_prediction.shape
    # #S = reformed_prediction.shape
    # print np.median(np.abs(reformed_prediction - vox.V[:S[0], :S[1], :S[2]]))

    # # now want to save this grid to disk and render using a video creator...
    # savepath = paths.Bricks.prediction % ('oracle', scenename)
    # oracle_prediction = vox.blank_copy()
    # oracle_prediction.V = brick_grid.to_voxel_grid()
    # oracle_prediction.save(savepath)
