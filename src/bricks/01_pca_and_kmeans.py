'''
first test with voxel bricks - aim is to divide up a voxel grid into bricls
'''

import sys
import os
import numpy as np
import cPickle as pickle

from sklearn.decomposition import RandomizedPCA
from sklearn.cluster import MiniBatchKMeans
from sklearn.cluster import KMeans

import bricks

sys.path.append(os.path.expanduser('~/projects/shape_sharing/src/'))

from common import voxel_data
from common import paths
from common import parameters

def pca_randomized(X_in, local_subsample_length, num_pca_dims):

    # take subsample
    rand_exs = np.sort(np.random.choice(
        X_in.shape[0],
        np.minimum(local_subsample_length, X_in.shape[0]),
        replace=False))
    X = X_in.take(rand_exs, 0)

    pca = RandomizedPCA(n_components=num_pca_dims)
    pca.fit(X)
    return pca


def cluster_data(X, local_subsample_length, num_clusters):

    # take subsample
    if local_subsample_length > X.shape[0]:
        X_subset = X
    else:
        to_use_for_clustering = \
            np.random.randint(0, X.shape[0], size=(local_subsample_length))
        X_subset = X[to_use_for_clustering, :]

    # doing clustering
    km = MiniBatchKMeans(
        n_clusters=num_clusters,
        batch_size=5,
        n_init=10)
    km.fit(X_subset)
    return km

brick_side = 10
all_brick_grids = []

print "Loading and dividing all scenes..."

for scenename in paths.RenderedData.get_scene_list():

    loadpath = paths.RenderedData.ground_truth_voxels(scenename)
    vox = voxel_data.load_voxels(loadpath)
    vox.V[np.isnan(vox.V)] = -parameters.RenderedVoxelGrid.mu

    brick_grid = bricks.Bricks()
    brick_grid.from_voxel_grid(vox.V, brick_side)
    this_scene_examples = brick_grid.to_flat()

    print this_scene_examples.shape
    all_brick_grids.append(this_scene_examples)

all_brick_grids_np = np.concatenate(all_brick_grids, axis=0)
print "All shape is ", all_brick_grids_np.shape

# removing bricks which are all empty ( should really do in some kind of stachistic manner)
to_remove = np.all(all_brick_grids_np > 0, axis=1)
all_brick_grids_np = all_brick_grids_np[~to_remove, :]
print "After filtering, all shape is ", all_brick_grids_np.shape

# now do the kmeans and pca on this data
# clustering the sboxes - but only a subsample of them for speed!
print "Doing PCA"
pca = pca_randomized(
    all_brick_grids_np,
    local_subsample_length=1e6,
    num_pca_dims=16)

print "Fitting data under the PCA model"
all_brick_grids_transformed = pca.transform(all_brick_grids_np)
print all_brick_grids_transformed.shape

print "Saving to " + paths.Bricks.pca
with open(paths.Bricks.pca, 'wb') as f:
    pickle.dump(pca, f, pickle.HIGHEST_PROTOCOL)


# print "Doing Kmeans"
# km = cluster_data(
#     all_brick_grids_transformed,
#     local_subsample_length=1e6,
#     num_clusters=500)

# print "Now saving..."

# print "Saving to " + paths.Bricks.kmeans
# with open(paths.Bricks.kmeans, 'wb') as f:
#     pickle.dump(km, f, pickle.HIGHEST_PROTOCOL)
