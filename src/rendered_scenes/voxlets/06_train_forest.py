'''
train the model dammit
'''
import numpy as np
import cPickle as pickle
import scipy.io
import sys, os
sys.path.append(os.path.expanduser('~/projects/shape_sharing/src/'))
from sklearn.ensemble import RandomForestClassifier

from common import paths

"Parameters"
if paths.host_name != 'troll':
    small_sample = True
    subsample_length = 1000
    number_trees = 10
    max_depth = 10
else:
    small_sample = False
    subsample_length = 1000000
    number_trees = 50
    max_depth = 10

if small_sample: print "WARNING: Just computing on a small sample"


####################################################################
print "Loading the dictionaries"
# load in the pca kmeans
km_pca = pickle.load(open(paths.voxlet_pca_dict_path, 'rb'))
km_standard = pickle.load(open(paths.voxlet_dict_path, 'rb'))

####################################################################
print "Loading in all the data..."
features = []
pca_kmeans_idx = []
pca_representation = []
kmeans_idx = []

for count, modelname in enumerate(paths.train_names):

    if modelname == 'nice_honey_roasted_almonds':
        continue
    # loading the data
    loadpath = paths.bigbird_training_data_fitted_mat % modelname
    print "Loading from " + loadpath

    D = scipy.io.loadmat(loadpath)

    print D['features'].shape
    features.append(D['features'])
    pca_kmeans_idx.append(D['pca_kmeans_idx'])
    kmeans_idx.append(D['kmeans_idx'])

    if count > 3 and small_sample:
        print "SMALL SAMPLE: Stopping"
        break

np_pca_kmeans_idx = np.hstack(pca_kmeans_idx).flatten()
np_kmeans_idx = np.hstack(kmeans_idx).flatten()


####################################################################
print "Now training the forest"
print np.array(features).shape
np_features = np.array(features).reshape((-1, 56))
to_remove = np.any(np.isnan(np_features), axis=1)

print ""
print "Before subsampling"
print "np_features has shape " + str(np_features.shape)
print "np_pca_kmeans_idx has shape " + str(np_pca_kmeans_idx.shape)
print "np_kmeans_idx has shape " + str(np_kmeans_idx.shape)

np_features = np_features[~to_remove, :]
np_pca_kmeans_idx = np_pca_kmeans_idx[~to_remove]
np_kmeans_idx = np_kmeans_idx[~to_remove]

print ""
print "Before subsampling"
print "np_features has shape " + str(np_features.shape)
print "np_pca_kmeans_idx has shape " + str(np_pca_kmeans_idx.shape)
print "np_kmeans_idx has shape " + str(np_kmeans_idx.shape)

rand_exs = np.sort(np.random.choice(np_features.shape[0], np.minimum(subsample_length, np_features.shape[0]), replace=False))
np_features = np_features.take(rand_exs, 0)
np_pca_kmeans_idx = np_pca_kmeans_idx.take(rand_exs, 0)
np_kmeans_idx = np_kmeans_idx.take(rand_exs, 0)

print ""
print "After subsampling"
print "np_features has shape " + str(np_features.shape)
print "np_pca_kmeans_idx has shape " + str(np_pca_kmeans_idx.shape)
print "np_kmeans_idx has shape " + str(np_kmeans_idx.shape)

forest_pca = RandomForestClassifier(n_estimators=number_trees, criterion="entropy", oob_score=True, max_depth=max_depth, n_jobs=8)
forest_pca.fit(np_features, np_pca_kmeans_idx)

print "Done training, now saving"
#forest_dict = dict(forest=forest, km_model=km_pca, pca_model=)
pickle.dump(forest_pca, open(paths.voxlet_model_pca_path, 'wb'))

forest = RandomForestClassifier(n_estimators=number_trees, criterion="entropy", oob_score=True, max_depth=max_depth, n_jobs=8)
forest.fit(np_features, np_kmeans_idx)

print "Done training, now saving"
#forest_dict = dict(forest=forest, pca_model=pca)
pickle.dump(forest, open(paths.voxlet_model_path, 'wb'))
