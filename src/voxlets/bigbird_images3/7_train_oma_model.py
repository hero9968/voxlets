'''
train the model dammit
'''
import numpy as np
import cPickle as pickle
import scipy.io
import sys, os
sys.path.append(os.path.expanduser('~/projects/shape_sharing/src/'))
from sklearn.ensemble import RandomForestClassifier
import time
from common import paths
import random_forest_structured as srf

"Parameters"
if paths.host_name != 'troll':
    small_sample = True
    subsample_length = 10000
    number_trees = 10
    max_depth = 10
else:
    small_sample = False
    subsample_length = 1000000
    number_trees = 100
    max_depth = 14
    
if small_sample: print "WARNING: Just computing on a small sample"


####################################################################
print "Loading the dictionaries etc"
# load in the pca components
pca = pickle.load(open(paths.voxlet_pca_path, 'rb'))


####################################################################
print "Loading in all the data..."
features = []
pca_kmeans_idx = []
pca_representation = []
kmeans_idx = []

for count, modelname in enumerate(paths.train_names):

    # loading the data
    loadpath = paths.bigbird_training_data_fitted_mat % modelname
    print "Loading from " + loadpath

    D = scipy.io.loadmat(loadpath)

    features.append(D['features'])
    
    pca_representation.append(D['pca_representation'])
    pca_kmeans_idx.append(D['pca_kmeans_idx'])
    kmeans_idx.append(D['kmeans_idx'])
    
    if count > 3 and small_sample:
        print "SMALL SAMPLE: Stopping"
        break

np_pca_kmeans_idx = np.hstack(pca_kmeans_idx).flatten()
np_pca_representation = np.vstack(pca_representation)
np_kmeans_idx = np.hstack(kmeans_idx).flatten()


####################################################################
print "Now training the forest"
np_features = np.array(features).reshape((-1, 56))
to_remove = np.any(np.isnan(np_features), axis=1)

print ""
print "Before subsampling"
print "np_features has shape " + str(np_features.shape)
print "np_pca_kmeans_idx has shape " + str(np_pca_kmeans_idx.shape)
print "np_pca_representation has shape " + str(np_pca_representation.shape)
print "np_kmeans_idx has shape " + str(np_kmeans_idx.shape)
print ""

np_features = np_features[~to_remove, :]
np_pca_kmeans_idx = np_pca_kmeans_idx[~to_remove]
np_pca_representation = np_pca_representation[~to_remove, :]
np_kmeans_idx = np_kmeans_idx[~to_remove]

print ""
print "Before subsampling"
print "np_features has shape " + str(np_features.shape)
print "np_pca_kmeans_idx has shape " + str(np_pca_kmeans_idx.shape)
print "np_pca_representation has shape " + str(np_pca_representation.shape)
print "np_kmeans_idx has shape " + str(np_kmeans_idx.shape)
print ""

rand_exs = np.sort(np.random.choice(np_features.shape[0], np.minimum(subsample_length, np_features.shape[0]), replace=False))
np_features = np_features.take(rand_exs, 0)
np_pca_kmeans_idx = np_pca_kmeans_idx.take(rand_exs, 0)
np_pca_representation = np_pca_representation.take(rand_exs, 0)
np_kmeans_idx = np_kmeans_idx.take(rand_exs, 0)

print ""
print "After subsampling"
print "np_features has shape " + str(np_features.shape)
print "np_pca_kmeans_idx has shape " + str(np_pca_kmeans_idx.shape)
print "np_pca_representation has shape " + str(np_pca_representation.shape)
print "np_kmeans_idx has shape " + str(np_kmeans_idx.shape)
print ""

forest_params = srf.ForestParams()
forest = srf.Forest(forest_params)
tic = time.time()
forest.train(np_features, np_pca_representation)
toc = time.time()
print 'train time', toc-tic

print "Combining forest with training data"
forest_dict = dict(forest=forest, traindata=np_pca_representation, pca_model=pca)

print "Done training, now saving"
pickle.dump(forest_dict, open(paths.voxlet_model_oma_path, 'wb'))
