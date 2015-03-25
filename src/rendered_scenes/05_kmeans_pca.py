import numpy as np
import cPickle as pickle
import scipy.io
import sys
import os
sys.path.append(os.path.expanduser('~/projects/shape_sharing/src/'))

from sklearn.decomposition import RandomizedPCA
from sklearn.cluster import MiniBatchKMeans
from sklearn.manifold import Isomap

from common import paths
from common import parameters
if parameters.small_sample:
    print "WARNING: Just computing on a small sample"


def pca_randomized(X_in, local_subsample_length, num_pca_dims):

    # take subsample
    rand_exs = np.sort(np.random.choice(
        X_in.shape[0],
        np.minimum(local_subsample_length, X_in.shape[0]),
        replace=False))
    X = X_in.take(rand_exs, 0)

    pca = RandomizedPCA(n_components=num_pca_dims)
    pca.fit(X)
    return pca


def do_isomap(X_in, local_subsample_length, num_isomap_dimensions):

    # take subsample
    rand_exs = np.sort(np.random.choice(
        X_in.shape[0],
        np.minimum(local_subsample_length, X_in.shape[0]),
        replace=False))
    X = X_in.take(rand_exs, 0)

    iso = Isomap(n_components=num_isomap_dimensions, n_neighbors=25)
    iso.fit(X.astype(np.float16))
    return iso


def cluster_data(X, local_subsample_length, num_clusters):

    # take subsample
    if local_subsample_length > X.shape[0]:
        X_subset = X
    else:
        to_use_for_clustering = \
            np.random.randint(0, X.shape[0], size=(local_subsample_length))
        X_subset = X[to_use_for_clustering, :]

    print X.shape
    print X_subset.shape

    # doing clustering
    km = MiniBatchKMeans(n_clusters=num_clusters)
    km.fit(X_subset)
    return km

# save path (open here so if an error is thrown I can catch it early...)

# initialise lists
shoeboxes = []
features = []

for count, sequence in enumerate(paths.RenderedData.train_sequence()[:80]):

    print "Processing " + sequence['name']

    # loading the data
    loadpath = paths.RenderedData.voxlets_dict_data_path + \
        sequence['name'] + '.mat'
    print "Loading from " + loadpath

    D = scipy.io.loadmat(loadpath)
    shoeboxes.append(D['shoeboxes'].astype(np.float16))
    features.append(D['features'].astype(np.float16))

    if count > parameters.max_sequences:
        print "SMALL SAMPLE: Stopping"
        break

np_all_sboxes = np.concatenate(shoeboxes, axis=0)
np_all_features = np.concatenate(features, axis=0)
print "All sboxes shape is " + str(np_all_sboxes.shape)
print "Features shape is " + str(np_all_features.shape)
print np_all_features.dtype
print np_all_sboxes.dtype

# Replacing nans with a low number in the features, hopefully will work...
np_all_features[np.isnan(np_all_features)] = -parameters.RenderedVoxelGrid.mu

for name, np_array in zip(
        ('shoeboxes', 'features'), (np_all_sboxes, np_all_features)):

    # clustering the sboxes - but only a subsample of them for speed!
    print "Doing PCA"
    pca = pca_randomized(
        np_array,
        parameters.VoxletTraining.pca_subsample_length,
        parameters.VoxletTraining.number_pca_dims)

    if name == 'features':
        # No point doing isomap for the shoeboxes
        print "Doing Isomapping"

        iso = do_isomap(
            np_array,
            parameters.VoxletTraining.pca_subsample_length / 10,
            parameters.VoxletTraining.number_pca_dims)

    print "Doing Kmeans"
    km = cluster_data(
        np_array,
        parameters.VoxletTraining.pca_subsample_length,
        parameters.VoxletTraining.number_clusters)

    try:
        pca_savepath = paths.RenderedData.voxlets_dictionary_path + name + \
            '_pca.pkl'
        kmeans_savepath = paths.RenderedData.voxlets_dictionary_path + name + \
            '_kmean.pkl'
        iso_savepath = paths.RenderedData.voxlets_dictionary_path + name + \
            '_iso.pkl'

        print "Saving to " + pca_savepath
        with open(pca_savepath, 'wb') as f:
            pickle.dump(pca, f, pickle.HIGHEST_PROTOCOL)

        print "Saving to " + kmeans_savepath
        with open(kmeans_savepath, 'wb') as f:
            pickle.dump(km, f, pickle.HIGHEST_PROTOCOL)

        if name == 'features':
            print "Saving to " + iso_savepath
            with open(iso_savepath, 'wb') as f:
                pickle.dump(iso, f, pickle.HIGHEST_PROTOCOL)

    except:
        import pdb; pdb.set_trace()
