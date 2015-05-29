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
from features import line_casting

parameters = yaml.load(open('./implicit_params.yaml'))


def train_model(model_params):

    all_X = []
    all_Y = []

    for sequence in paths.all_train_data:

        # loading the data and adding to arrays
        print "Loading from %s" % sequence['name']
        file_to_load = paths.implicit_training_dir % parameters['features_name'] + sequence['name'] + '.mat'
        if os.path.exists(file_to_load):
            training_pair = scipy.io.loadmat(file_to_load)
        else:
            print "WARNING - could not find file ", file_to_load

        training_pair['rays'] = line_casting.postprocess_features(
            training_pair['rays'], model_params['postprocess'])

        feats = np.hstack(
            [training_pair[feat_name] for feat_name in model_params['features']])

        print "Features shape is ", feats.shape, training_pair['Y'].shape

        all_X.append(feats.astype(np.float32))
        all_Y.append(training_pair['Y'].astype(np.float16))

    all_X_np = np.concatenate(all_X, axis=0).astype(np.float32)
    all_Y_np = np.concatenate(all_Y, axis=1).flatten().astype(np.float16)

    max_pairs = model_params['forest']['max_training_pairs']
    if all_X_np.shape[0] > max_pairs:
        print "Resampling %d pairs to %d pairs" % (all_X_np.shape[0], max_pairs)
        idxs = np.random.choice(all_X_np.shape[0], max_pairs)
        all_X_np = all_X_np[idxs, :]
        all_Y_np = all_Y_np[idxs]

    print all_X_np.shape, all_Y_np.shape
    print all_X_np.dtype, all_Y_np.dtype

    print "Training the model"
    rf = sklearn.ensemble.RandomForestRegressor(
        n_estimators=model_params['forest']['ntrees'],
        oob_score=True,
        n_jobs=system_setup.cores,
        max_depth=model_params['forest']['max_depth'],
        max_features=all_X_np.shape[1])
    rf.fit(all_X_np, all_Y_np)

    # adding additional parameters to the model
    rf.parameters = model_params

    return rf


if __name__ == '__main__':
    for model_params in parameters['models']:

        print "Training model", model_params['name']
        rf = train_model(model_params)

        print "Creating the save location"
        savefolder = paths.implicit_model_dir % model_params['name']
        if not os.path.exists(savefolder):
            os.makedirs(savefolder)

        print "Saving the model"
        savepath = savefolder + 'model.pkl'
        with open(savepath, 'wb') as f:
            pickle.dump(rf, f, protocol=pickle.HIGHEST_PROTOCOL)
