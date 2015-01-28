'''
train the model dammit
'''
import numpy as np
import cPickle as pickle
import scipy.io
import sys
import os
import time

sys.path.append(os.path.expanduser('~/projects/shape_sharing/src/'))
from common import paths
from common import random_forest_structured as srf

if paths.small_sample:
    print "WARNING: Just computing on a small sample"

subsample_length = 10000

####################################################################
print "Loading the dictionaries etc"
########################################################################
pca_savepath = paths.RenderedData.voxlets_dictionary_path + 'pca.pkl'
with open(pca_savepath, 'rb') as f:
    pca = pickle.load(f)

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

    pca_representation.append(D['pca_representation'])
    # pca_kmeans_idx.append(D['pca_kmeans_idx'])
    # kmeans_idx.append(D['kmeans_idx'])

    if count > 8 and paths.small_sample:
        print "SMALL SAMPLE: Stopping"
        break

np_pca_representation = np.vstack(pca_representation)
# np_kmeans_idx = np.hstack(kmeans_idx).flatten()

####################################################################
print "Subsampling"
########################################################################

# np_features = np.array(features).reshape((-1, 56))

np_features = np.concatenate(features, axis=0)
to_remove = np.any(np.isnan(np_features), axis=1)

print "Before subsampling"
print "np_features has shape " + str(np_features.shape)
print "np_pca_representation has shape " + str(np_pca_representation.shape)

np_features = np_features[~to_remove, :]
np_pca_representation = np_pca_representation[~to_remove, :]

print "Before subsampling"
print "np_features has shape " + str(np_features.shape)
print "np_pca_representation has shape " + str(np_pca_representation.shape)

rand_exs = np.sort(np.random.choice(
    np_features.shape[0],
    np.minimum(subsample_length, np_features.shape[0]),
    replace=False))
np_features = np_features.take(rand_exs, 0)
np_pca_representation = np_pca_representation.take(rand_exs, 0)

print "After subsampling"
print "np_features has shape " + str(np_features.shape)
print "np_pca_representation has shape " + str(np_pca_representation.shape)

########################################################################
print "Training OMA forest"
########################################################################

forest_params = srf.ForestParams()
forest = srf.Forest(forest_params)
tic = time.time()
forest.train(np_features, np_pca_representation)
toc = time.time()
print 'train time', toc-tic

print "Combining forest with training data"
forest_dict = dict(
    forest=forest,
    traindata=np_pca_representation,
    pca_model=pca)

print "Done training, now saving"
pickle.dump(forest_dict, open(paths.voxlet_model_oma_path, 'wb'))
