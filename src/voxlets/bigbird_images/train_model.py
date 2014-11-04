'''
loads all the shoeboxes and features and classifies each sbox against the dictionary centres
'''

import numpy as np
import cPickle as pickle
import sys, os
sys.path.append(os.expanduser('~/projects/shape_sharing/src/'))
import scipy.io

from sklearn.ensemble import RandomForestClassifier

from common import paths

'''PARAMETERS'''
fv_dimension = 56

all_features = []
all_idxs = []


print "Starting main loop"
for modelname in paths.train_names[:3]:

    # loading in the combined features for this model
    load_path = paths.base_path + "voxlets/bigbird/" + modelname + "/combined.mat"
    print "Loading from " + load_path
    combined = pickle.load(open(load_path, 'rb'))

    # add to the main list
    all_features.append(combined['features'])
    all_idxs.append(combined['idxs'])

    print "Loaded " + modelname

print "All shape is " + str(np.array(all_features).shape)
np_all_features = np.array(all_features).reshape((-1, fv_dimension))
# let's remove rows which have any nans - these must be from outside the mask and so a bit dodge
to_remove = np.any(np.isnan(np_all_features), axis=1)
np_all_features = np_all_features[~to_remove, :]
print "All features is " + str(np_all_features.shape)

np_all_idxs = np.array(all_idxs).flatten()[~to_remove]
print "All idxs is " + str(np_all_idxs.shape)

print "Now training the model"
forest = RandomForestClassifier(n_estimators=50, criterion="entropy", oob_score=True)
forest.fit(np_all_features, np_all_idxs)

print "Saving forest"
save_folder = paths.base_path + "voxlets/models/"
if not os.path.exists(save_folder):
    os.mkdir(save_folder)
save_path = save_folder + "rf.pkl"
pickle.dump(forest, open(save_path, 'wb'))
