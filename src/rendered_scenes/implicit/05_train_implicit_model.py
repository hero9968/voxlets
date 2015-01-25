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
import random

max_training_pairs = 1e6

sys.path.append(os.path.expanduser('~/projects/shape_sharing/src/'))
from common import paths

with open(paths.yaml_train_location, 'r') as f:
    train_sequences = yaml.load(f)

all_X = []
all_Y = []

for sequence in train_sequences:

    # loading the data and adding to arrays
    print "Loading from %s" % sequence['name']
    seq_foldername = paths.sequences_save_location + sequence['name'] + '/'
    training_pair = scipy.io.loadmat(seq_foldername + 'training_pairs.mat')

    all_X.append(training_pair['X'].astype(np.float32))
    all_Y.append(training_pair['Y'].astype(np.float16))

all_X_np = np.concatenate(all_X, axis=0).astype(np.float32)
all_Y_np = np.concatenate(all_Y, axis=1).flatten().astype(np.float16)

if all_X_np.shape[0] > max_training_pairs:
    print "Resampling %d pairs to %d pairs" % \
        (all_X_np.shape[0], max_training_pairs)
    idxs = random.sample(range(all_X_np.shape[0]), max_training_pairs)
    all_X_np = all_X_np[idxs, :]
    all_Y_np = all_Y_np[idxs]

print all_X_np.shape
print all_Y_np.shape
print all_X_np.dtype
print all_Y_np.dtype

print "Training the model"
rf = sklearn.ensemble.RandomForestRegressor(
    n_estimators=50, oob_score=True, n_jobs=4, max_depth=14)
rf.fit(all_X_np, all_Y_np)

print "Saving the model"
pickle.dump(rf, open(paths.implicit_models_folder + 'model.pkl', 'wb'))
