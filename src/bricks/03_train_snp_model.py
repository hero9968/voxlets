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

    brick_grid = bricks.divide_up_voxel_grid(vox.V, brick_side)

    for x_slice in brick_grid:
        for y_slice in x_slice:

            # extract a column of bricks from this scene and pca transform
            col_of_bricks = y_slice.reshape((7, 1000))
            col_of_bricks_pca = pca.transform(col_of_bricks)

            # form pairs of adjacient ones...
            for idx in range(col_of_bricks_pca.shape[0]-1):
                # only add if the zero level set goes through this brick
                if np.any(col_of_bricks[idx+1] > 0) and np.any(col_of_bricks[idx+1] < 0):
                    training_X.append(col_of_bricks_pca[idx])
                    training_Y.append(col_of_bricks_pca[idx+1])

    print len(training_X)
    print len(training_Y)

print "Now learning the pairwise relationships..."
print "Let's do this in the z direction for now..."
print "Using OMA forest..."

X = np.vstack(training_X).astype(np.float32)
Y = np.vstack(training_Y).astype(np.float32)
print X.shape, Y.shape

forest_params = srf.ForestParams()
forest = srf.Forest(forest_params)
tic = time.time()
forest.train(X, Y)
print "Forest took %fs to train" % (time.time() - tic)

forest_save_path = paths.Bricks.models + 'z_dir_offset_1.pkl'

print "Saving to " + forest_save_path
with open(forest_save_path, 'wb') as f:
    pickle.dump(forest, f, pickle.HIGHEST_PROTOCOL)
