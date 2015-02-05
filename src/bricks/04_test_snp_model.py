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
from common import mesh
from common import parameters
from common import random_forest_structured as srf

brick_side = 10
percent_to_remove = 5.0

offset = [0, 0, 1]
forest_save_path = paths.Bricks.models + 'z_dir_offset_1.pkl'
with open(paths.Bricks.pca, 'rb') as f:
    pca = pickle.load(f)
with open(forest_save_path, 'rb') as f:
    rf = pickle.load(f)

print "Loading voxels"
scenename = paths.RenderedData.get_scene_list()[0]

loadpath = paths.RenderedData.ground_truth_voxels(scenename)
vox = voxel_data.load_voxels(loadpath)
vox.V[np.isnan(vox.V)] = -parameters.RenderedVoxelGrid.mu

print "Converting to bricks"
bgrid = bricks.Bricks()
bgrid.from_voxel_grid(vox.V, brick_side)

print "Removing %f percent of surface blocks" % percent_to_remove
removed_idxs = []
S = bgrid.B.shape[:3]
for i in range(S[0]):
    for j in range(S[1]):
        for k in range(S[2]):
            temp = bgrid.B[i, j, k]
            if np.any(temp > 0) and np.any(temp < 0):
                if np.random.rand() < percent_to_remove/100.0:
                    temp *= 0
                    temp += parameters.RenderedVoxelGrid.mu
                    removed_idxs.append([i, j, k])

print "Saving this modified voxel grid"
snp_savepath = paths.Bricks.prediction % ('snp_noise', scenename)
brick_vox = vox.blank_copy()
brick_vox.V = bgrid.to_voxel_grid()
brick_vox.save(snp_savepath)

# print brick_vox.V.shape
# ms = mesh.Mesh()
# ms.from_volume(brick_vox, 0.0)
# ms.write_to_obj('data/shit.obj')

print "Attempting to fill in the holes"
for (i, j, k) in removed_idxs:
    print bgrid.B[i, j, k].shape

    # get the feature vector which will be used to make the prediction for this brick
    source_i = i - offset[0]
    source_j = j - offset[1]
    source_k = k - offset[2]
    source_X = bgrid.B[source_i, source_j, source_k].flatten()

    prediction = pca.inverse_transform(rf.predict(pca.transform(source_X)))
    bgrid.B[i, j, k] = prediction.reshape((brick_side, brick_side, brick_side))


snp_savepath = paths.Bricks.prediction % ('snp_prediction', scenename)
brick_vox = vox.blank_copy()
brick_vox.V = bgrid.to_voxel_grid()
brick_vox.save(snp_savepath)

print "Should finally try to render these or something..."
brick_vox.render(snp_frame_savepath)
