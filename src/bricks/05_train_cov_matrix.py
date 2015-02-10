'''
doing a partial lookup of each z-brick to a training set, aiming to find out
its nearest neighbour.
Option:
Could do:
    1) full lookup to training set, OR
    2) partial lookup to test set

Full lookup has more training examples but because dimensions are fixed we can
train a forest or something
'''
import sys
import os
import numpy as np
import cPickle as pickle

import bricks
sys.path.append(os.path.expanduser('~/projects/shape_sharing/src/'))
from common import voxel_data
from common import paths
from common import parameters

brick_side = brick_side = (10, 10, 75)

print "Training..."

all_bricks = []

for count, sequence in enumerate(paths.RenderedData.train_sequence()):

    print "Processing " + sequence['name']

    # load in the ground truth grid for this scene, and converting nans
    gt_vox = voxel_data.load_voxels(
        paths.RenderedData.ground_truth_voxels(sequence['scene']))
    gt_vox.V[np.isnan(gt_vox.V)] = -parameters.RenderedVoxelGrid.mu
    gt_vox.set_origin(gt_vox.origin)

    brick_grid = bricks.Bricks()
    brick_grid.from_voxel_grid(gt_vox.V, brick_side)
    all_bricks.append(brick_grid.to_flat())

all_bricks_np = np.vstack(all_bricks).astype(np.float32)
print all_bricks_np.shape

print "Saving to disk"
with open('data/training_data.pkl', 'wb') as f:
    pickle.dump(all_bricks_np, f, protocol=pickle.HIGHEST_PROTOCOL)

quit()

print "Performing covariance"
cov = np.cov(all_bricks_np.T).astype(np.float32)
mean = np.mean(all_bricks_np, axis=0).astype(np.float32)
print cov.shape

with open('data/cov.pkl', 'wb') as f:
    pickle.dump([mean, cov], f, protocol=pickle.HIGHEST_PROTOCOL)

quit()

