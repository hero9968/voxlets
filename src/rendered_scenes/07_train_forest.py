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

masks_pca_savepath = paths.RenderedData.voxlets_dictionary_path + 'masks_pca.pkl'
with open(features_pca_savepath, 'rb') as f:
    masks_pca = pickle.load(f)

####################################################################
print "Loading in all the data..."
########################################################################
features = []
pca_representation = []
masks = []
scene_ids = []

for count, sequence in enumerate(paths.RenderedData.train_sequence()):

    # loading the data
    loadpath = paths.RenderedData.voxlets_data_path + \
        sequence['name'] + '.mat'
    print "Loading from " + loadpath

    D = scipy.io.loadmat(loadpath)
    features.append(D['features'])
    pca_representation.append(D['shoeboxes'])
    masks.append(D['masks'])
    scene_ids.append(np.ones(D['features'].shape[0]) * count)

    print D['features'].shape, D['shoeboxes'].shape
    if count > parameters.max_sequences:
        print "SMALL SAMPLE: Stopping"
        break

np_pca_representation = np.vstack(pca_representation)
np_masks = np.vstack(masks)
np_features = np.concatenate(features, axis=0)
np_scene_ids = np.concatenate(scene_ids, axis=0).astype(int)

print "Sbox pca representation is shape", np_pca_representation.shape
print "Masks is ", np_masks.shape
print "Features is ", np_features.shape
print "Scene ids is ", np_scene_ids.shape

####################################################################
print "Training the model"
####################################################################

model = voxlets.VoxletPredictor()
model.set_voxlet_params(parameters.Voxlet)
model.train(
    np_features,
    np_pca_representation,
    parameters.VoxletTraining.forest_subsample_length,
    masks=np_masks,
    scene_ids=np_scene_ids)
model.set_pca(pca)
model.set_masks_pca(masks_pca)
model.set_feature_pca(features_pca)
model.save(paths.RenderedData.voxlet_model_oma_path)
