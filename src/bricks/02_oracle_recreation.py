'''
Aim is to use an oracle to replace each brick in a scene with the cloest match
from the kmeans clusters or similar...
'''
import sys
import os
import numpy as np
import cPickle as pickle

from sklearn.decomposition import RandomizedPCA
from sklearn.cluster import MiniBatchKMeans

import bricks
sys.path.append(os.path.expanduser('~/projects/shape_sharing/src/'))
from common import voxel_data
from common import paths
from common import parameters

brick_side = 75
oracle_type = 'pca'

with open(paths.Bricks.pca, 'rb') as f:
    pca = pickle.load(f)

with open(paths.Bricks.kmeans, 'rb') as f:
    km = pickle.load(f)

# load in a scene and divide it...
for scenename in paths.RenderedData.get_scene_list():

    print "Working on " + scenename

    loadpath = paths.RenderedData.ground_truth_voxels(scenename)
    vox = voxel_data.load_voxels(loadpath)
    vox.V[np.isnan(vox.V)] = -parameters.RenderedVoxelGrid.mu

    brick_grid = bricks.Bricks()
    brick_grid.from_voxel_grid(vox.V, brick_side)
    this_scene_examples = brick_grid.to_flat()

    pca_bricks = pca.transform(this_scene_examples)
    if oracle_type == 'kmeans':
        cluster_idxs = km.predict(pca_bricks)
        closest_clusters_pca = km.cluster_centers_[cluster_idxs]
        closest_clusters = pca.inverse_transform(closest_clusters_pca)
    elif oracle_type == 'pca':
        closest_clusters = pca.inverse_transform(pca_bricks)

    brick_grid.from_flat(closest_clusters)
        # closest_clusters,
        # brick_grid.shape[:3],
        # brick_side=brick_side,
        # original_shape=vox.V.shape)

    #print reformed_prediction.shape
    #S = reformed_prediction.shape
    #print np.median(np.abs(reformed_prediction - vox.V[:S[0], :S[1], :S[2]]))

    # now want to save this grid to disk and render using a video creator...
    savepath = paths.Bricks.prediction % ('oracle', scenename)
    oracle_prediction = vox.blank_copy()
    oracle_prediction.V = brick_grid.to_voxel_grid()
    oracle_prediction.save(savepath)
