'''
training a model to recover from salt and pepper noise,
i.e. if some of the bricks are removed, try to predict them...

currently just predicts upwards in the z direction
'''

import sys
import os
import numpy as np
import cPickle as pickle
import time
import bricks

sys.path.append(os.path.expanduser('~/projects/shape_sharing/src/'))

from common import voxel_data
from common import paths
from common import parameters
from common import random_forest_structured as srf

brick_side = 10

with open(paths.Bricks.pca, 'rb') as f:
    pca = pickle.load(f)

print "Loading all the scenes"
all_brick_grids = []

# this will store pairs of pca representations of voxel grids...
training_X = []
training_Y = []

for scenename in paths.RenderedData.get_scene_list():

    loadpath = paths.RenderedData.ground_truth_voxels(scenename)
    vox = voxel_data.load_voxels(loadpath)
    vox.V[np.isnan(vox.V)] = -parameters.RenderedVoxelGrid.mu

    bgrid = bricks.Bricks()
    bgrid.from_voxel_grid(vox.V, brick_side)

    X, Y = bgrid.get_adjacient_bricks([0, 0, 1])

    training_X.append(X)
    training_Y.append(Y)

    print len(training_X)
    print len(training_Y)

print "Now learning the pairwise relationships..."
print "Let's do this in the z direction for now..."
print "Using OMA forest..."

X = np.vstack(training_X).astype(np.float32)
Y = np.vstack(training_Y).astype(np.float32)
X = X.reshape((X.shape[0], -1))
Y = Y.reshape((Y.shape[0], -1))
X = pca.transform(X)
Y = pca.transform(Y)

print X.shape, Y.shape

forest_params = srf.ForestParams()
forest = srf.Forest(forest_params)
tic = time.time()
forest.train(X, Y)
forest.data = Y
print "Forest took %fs to train" % (time.time() - tic)

forest_save_path = paths.Bricks.models + 'z_dir_offset_1.pkl'

print "Saving to " + forest_save_path
with open(forest_save_path, 'wb') as f:
    pickle.dump(forest, f, pickle.HIGHEST_PROTOCOL)
