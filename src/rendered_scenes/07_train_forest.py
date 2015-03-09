'''
train the model dammit
'''
import numpy as np
import scipy.io
import cPickle as pickle
import sys
import os
sys.path.append(os.path.expanduser('~/projects/shape_sharing/src/'))
from common import paths
from common import parameters
from common import voxlets
import time

if parameters.small_sample:
    print "WARNING: Just computing on a small sample"

modelfolder = os.path.dirname(paths.RenderedData.voxlet_model_oma_path)
if not os.path.exists(modelfolder):
    os.makedirs(modelfolder)


####################################################################
print "Loading the dictionaries and PCA"
########################################################################
pca_savepath = paths.RenderedData.voxlets_dictionary_path + 'shoeboxes_pca.pkl'
with open(pca_savepath, 'rb') as f:
    pca = pickle.load(f)

features_pca_savepath = paths.RenderedData.voxlets_dictionary_path + 'features_pca.pkl'
with open(features_pca_savepath, 'rb') as f:
    features_pca = pickle.load(f)

####################################################################
print "Loading in all the data..."
########################################################################
features = []
pca_representation = []

for count, sequence in enumerate(paths.RenderedData.train_sequence()):

    # loading the data
    loadpath = paths.RenderedData.voxlets_data_path + \
        sequence['name'] + '.mat'
    print "Loading from " + loadpath

    D = scipy.io.loadmat(loadpath)
    features.append(D['features'])
    pca_representation.append(D['shoeboxes'])

    print D['features'].shape, D['shoeboxes'].shape
    if count > parameters.max_sequences:
        print "SMALL SAMPLE: Stopping"
        break

np_pca_representation = np.vstack(pca_representation)
np_features = np.concatenate(features, axis=0)

print np_pca_representation.shape
print np_features.shape

####################################################################
print "Training the model"
####################################################################

model = voxlets.VoxletPredictor()
model.train(
    np_features,
    np_pca_representation,
    parameters.VoxletTraining.forest_subsample_length)
model.set_pca(pca)
model.set_feature_pca(features_pca)
model.save(paths.RenderedData.voxlet_model_oma_path)
