'''
train the model dammit
'''
import numpy as np
import scipy.io
import cPickle as pickle
import sys
import os
sys.path.append(os.path.expanduser('~/projects/shape_sharing/src/'))
import real_data_paths as paths
import real_params as parameters
from common import voxlets
import time

if parameters.small_sample:
    print "WARNING: Just computing on a small sample"

modelfolder = os.path.dirname(paths.voxlet_model_oma_path)
if not os.path.exists(modelfolder):
    os.makedirs(modelfolder)


####################################################################
print "Loading the dictionaries and PCA"
########################################################################
pca_savepath = paths.voxlets_dictionary_path + 'shoeboxes_pca.pkl'
with open(pca_savepath, 'rb') as f:
    pca = pickle.load(f)

####################################################################
print "Loading in all the data..."
########################################################################
features = []
pca_representation = []
scene_ids = []



for count, sequence in enumerate(paths.train_data):

    # loading the data
    loadpath = paths.voxlets_data_path + \
        sequence['name'] + '.pkl'

    try:
        with open(loadpath, 'r') as f:
            D = pickle.load(f)
        features.append(D['cobweb'])
        pca_representation.append(D['shoeboxes'])
        scene_ids.append(np.ones(D['cobweb'].shape[0]) * count)
    except:
        print "Failed to loading from " + loadpath
        continue


np_pca_representation = np.vstack(pca_representation)
np_features = np.concatenate(features, axis=0)
np_scene_ids = np.concatenate(scene_ids, axis=0).astype(int)

print "Sbox pca representation is shape", np_pca_representation.shape
print "Features is ", np_features.shape
print "Scene ids is ", np_scene_ids.shape

# print np_scene_ids
large_value = -5.0
np_features[np.isnan(np_features)] = large_value

####################################################################
print "Training the model"
####################################################################

model = voxlets.VoxletPredictor()
model.set_voxlet_params(parameters.Voxlet)
model.train(
    np_features,
    np_pca_representation,
    parameters.VoxletTraining.forest_subsample_length,
    scene_ids=np_scene_ids)
model.set_pca(pca)
model.save(paths.voxlet_model_oma_path.replace('.pkl', '_cobweb.pkl'))
