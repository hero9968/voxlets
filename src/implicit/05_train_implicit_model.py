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
import yaml

sys.path.append(os.path.expanduser('~/projects/shape_sharing/src/'))

import system_setup
import real_data_paths as paths

parameters = yaml.load(open('./implicit_params.yaml'))
modelname = parameters['modelname']

print "Creating the save location"
savefolder = paths.implicit_model_dir % modelname
if not os.path.exists(savefolder):
    os.makedirs(savefolder)

all_X = []
all_Y = []

for sequence in paths.all_train_data:

    # loading the data and adding to arrays
    print "Loading from %s" % sequence['name']
    file_to_load = paths.implicit_training_dir % modelname + sequence['name'] + '.mat'
    if os.path.exists(file_to_load):
        training_pair = scipy.io.loadmat(file_to_load)
    else:
        print "WARNING - could not find file ", file_to_load

    feats = np.hstack(
        [training_pair[feat_name] for feat_name in parameters['features']])
    print "Features shape is ", feats.shape, training_pair['Y'].shape

    all_X.append(feats.astype(np.float32))
    all_Y.append(training_pair['Y'].astype(np.float16))

all_X_np = np.concatenate(all_X, axis=0).astype(np.float32)
all_Y_np = np.concatenate(all_Y, axis=1).flatten().astype(np.float16)

if all_X_np.shape[0] > parameters['max_training_pairs']:
    print "Resampling %d pairs to %d pairs" % \
        (all_X_np.shape[0], parameters['max_training_pairs'])
    idxs = np.random.choice(all_X_np.shape[0], parameters['max_training_pairs'])
    all_X_np = all_X_np[idxs, :]
    all_Y_np = all_Y_np[idxs]

print all_X_np.shape, all_Y_np.shape
print all_X_np.dtype, all_Y_np.dtype

print "Training the model"
rf = sklearn.ensemble.RandomForestRegressor(
    n_estimators=parameters['forest']['ntrees'],
    oob_score=True,
    n_jobs=system_setup.cores,
    max_depth=parameters['forest']['max_depth'],
    max_features=parameters['forest']['max_features'])
rf.fit(all_X_np, all_Y_np)

# adding additional parameters to the model
rf.parameters = parameters

print "Saving the model"
with open(savefolder + 'model.pkl', 'wb') as f:
    pickle.dump(rf, f, protocol=pickle.HIGHEST_PROTOCOL)
