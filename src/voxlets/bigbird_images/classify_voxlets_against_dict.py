'''
loads all the shoeboxes and features and classifies each sbox against the dictionary centres
'''

import numpy as np
import cPickle as pickle
import sys, os
sys.path.append(os.path.expanduser('~/projects/shape_sharing/src/'))
import scipy.io

from common import voxel_data
from common import paths
from common import mesh
from common import images
from common import features

from shoebox_helpers import *

'''PARAMETERS'''
overwrite = False
fv_dimension = 56

print "Loading in the dictionary"
shoebox_kmeans_path = paths.base_path + "voxlets/dict/clusters_from_images.pkl"
km = pickle.load(open(shoebox_kmeans_path, 'rb'))


print "Starting main loop"
for modelname in paths.modelnames:

    model_features = []
    model_sbox_idxs = []

    # skipping this model if it seems we've already saved all the required files       
    save_path = paths.base_path + "voxlets/bigbird/" + modelname + "/combined.mat"
    if os.path.exists(save_path):
        print "Skipping model " + modelname
        continue
    
    for view in paths.views:

        # loading the features and sboxes for this model
        try:
            print "Loading in the shoeboxes for " + modelname + " " + view
            sbox, cobwebs, spiders = load_bigbird_shoeboxes_and_features(modelname, view)
            combined_features = np.concatenate((cobwebs, spiders), axis=1)

            print "Classifying sboxes and adding to list"
            idxs = km.predict(sbox)

            model_features.append(combined_features)
            model_sbox_idxs.append(idxs)

        except:
            print "Failed to load " + modelname + " " + view

    # combine all for this model
    np_features = np.array(model_features).reshape((-1, fv_dimension))
    np_idxs = np.array(model_sbox_idxs).flatten()
    print np_features.shape
    print np_idxs.shape

    combined = dict(features=np_features, idxs=np_idxs, dict="voxlets/dict/clusters_from_images.pkl")

    print "Saving for this model"
    #pickle.dump(combined, open(save_path, 'wb'))
    pickle.dump(combined, open(save_path, 'wb'))


