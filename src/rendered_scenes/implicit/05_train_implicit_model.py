'''
script to train a model
based on training data
'''
import sys, os
import numpy as np
import scipy.io
import sklearn.ensemble
import cPickle as pickle
import random

max_training_pairs = int(1e6)

sys.path.append(os.path.expanduser('~/projects/shape_sharing/src/'))
from common import paths
from common import parameters

all_X = []
all_Y = []

for sequence in paths.RenderedData.train_sequence():

    # loading the data and adding to arrays
    print "Loading from %s" % sequence['name']
    seq_foldername = paths.RenderedData.implicit_training_dir % sequence['name']
    file_to_load = seq_foldername + 'training_pairs.mat'
    if os.path.exists(file_to_load):
        training_pair = scipy.io.loadmat(file_to_load)
    else:
        print "WARNING - could not find file ", file_to_load

    all_X.append(training_pair['X'].astype(np.float32))
    all_Y.append(training_pair['Y'].astype(np.float16))

all_X_np = np.concatenate(all_X, axis=0).astype(np.float32)
all_Y_np = np.concatenate(all_Y, axis=1).flatten().astype(np.float16)

if all_X_np.shape[0] > max_training_pairs:
    print "Resampling %d pairs to %d pairs" % \
        (all_X_np.shape[0], max_training_pairs)
    idxs = random.sample(xrange(all_X_np.shape[0]), max_training_pairs)
    all_X_np = all_X_np[idxs, :]
    all_Y_np = all_Y_np[idxs]

print all_X_np.shape
print all_Y_np.shape
print all_X_np.dtype
print all_Y_np.dtype

print "Training the model"
rf = sklearn.ensemble.RandomForestRegressor(
    n_estimators=100, oob_score=True, n_jobs=parameters.cores, max_depth=12)
rf.fit(all_X_np, all_Y_np)

print "Saving the model"
with open(paths.RenderedData.implicit_models_dir + 'model.pkl', 'wb') as f:
    pickle.dump(rf, f, protocol=pickle.HIGHEST_PROTOCOL)
