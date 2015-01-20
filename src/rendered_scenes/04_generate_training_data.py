'''
script to train a model
based on training data
'''
import sys, os
import numpy as np
import yaml
import scipy.io
import sklearn.ensemble
import cPickle as pickle

sys.path.append(os.path.expanduser('~/projects/shape_sharing/src/'))
sys.path.append(os.path.expanduser('~/projects/shape_sharing/src/intrinsic/'))
from common import paths
from common import voxel_data
from common import carving
from features import line_casting

print "Computing training data for %d scenes" % \
    len(paths.rendered_primitive_scenes)
scene_names_to_use = paths.rendered_primitive_scenes#[:num_scenes_to_use]

for scene_name in ['YP8G55G2GZ']: #scene_names_to_use:

    input_data_path = paths.scenes_location + scene_name

    print "Loading the ground truth"
    gt_vox = voxel_data.load_voxels(input_data_path + '/voxelgrid.pkl')
    gt_vox.V[np.isnan(gt_vox.V)] = -0.1
    visible = voxel_data.load_voxels(input_data_path + '/visiblegrid.pkl')

    # getting the known full and empty voxels based on the depth image
    known_full_voxels = visible
    known_empty_voxels = gt_vox.blank_copy()
    known_empty_voxels.V = gt_vox.V > 0

    print "Computing features"
    X, Y = line_casting.feature_pairs_3d(
        known_empty_voxels, known_full_voxels, gt_vox.V,
        samples=50000, base_height=0, autorotate=False, all_voxels=True)

    print "Saving"
    training_pairs = dict(X=X, Y=Y,
        known_full=known_full_voxels.V.astype(np.float32),
        known_empty=known_empty_voxels.V.astype(np.float32),
        gt_vox=gt_vox.V.astype(np.float32))
    scipy.io.savemat(input_data_path + '/training_pairs.mat', training_pairs)
