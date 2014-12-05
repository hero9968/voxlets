'''
training the model for implicit voxel occupancy
'''

import numpy as np
import scipy.io
import sklearn.ensemble
import cPickle as pickle

import sys, os
sys.path.append(os.path.expanduser('~/projects/shape_sharing/src/'))
from common import paths

load_path_template = paths.base_path + '/implicit/bigbird/training_data/%s_%s.mat'
rf_save_path = paths.base_path + '/implicit/bigbird/rf/rf_shallow.pkl'

all_X = []
all_Y = []

# Loading the data
for modelname in paths.train_names:

    for view_idx in paths.views[:45]:

        # loading in the training data
        load_name = load_path_template % (modelname, view_idx)
        D = scipy.io.loadmat(load_name)

        # adding to the full set
        all_X.append(D['X'])
        all_Y.append(D['Y'])


    print "Done model " + modelname

all_X_np = np.concatenate(all_X, axis=0)
all_Y_np = np.concatenate(all_Y, axis=1).flatten().astype(np.float16)

print all_X_np.shape
print all_Y_np.shape
print all_X_np.dtype
print all_Y_np.dtype

print "Training the model"
rf = sklearn.ensemble.RandomForestRegressor(n_estimators=50, oob_score=True, n_jobs=4, max_depth=12)
rf.fit(all_X_np, all_Y_np)

print "Saving the model"
pickle.dump(rf, open(rf_save_path, 'wb'))