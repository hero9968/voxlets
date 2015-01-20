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

print len(paths.rendered_primitive_scenes)
scene_names_to_use = paths.rendered_primitive_scenes#[:num_scenes_to_use]

all_X = []
all_Y = []

for scene_name in scene_names_to_use:

    input_data_path = paths.scenes_location + scene_name

    print "Loading data for %s" % scene_name
    training_pair = scipy.io.loadmat(input_data_path + '/training_pairs.mat')

    all_X.append(training_pair['X'])
    all_Y.append(training_pair['Y'])

all_X_np = np.concatenate(all_X, axis=0).astype(np.float32)
all_Y_np = np.concatenate(all_Y, axis=1).flatten().astype(np.float16)

print all_X_np.shape
print all_Y_np.shape
print all_X_np.dtype
print all_Y_np.dtype

print "Training the model"
rf = sklearn.ensemble.RandomForestRegressor(
    n_estimators=50, oob_score=True, n_jobs=4, max_depth=14)
rf.fit(all_X_np, all_Y_np)

print "Saving the model"
with open(paths.implicit_models_folder + 'full_scan_model.pkl', 'wb') as f:
    pickle.dump(rf, f)