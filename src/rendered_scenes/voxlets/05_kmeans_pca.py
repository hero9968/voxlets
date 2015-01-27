import numpy as np
import cPickle as pickle
import scipy.io
import sys
import os
sys.path.append(os.path.expanduser('~/projects/shape_sharing/src/'))

from sklearn.decomposition import RandomizedPCA
from sklearn.cluster import MiniBatchKMeans

from common import paths
from common import parameters

# parameters
subsample_length = 25000

# how many points to cluster with
number_pca_dims = 50
number_clusters = 30
if parameters.small_sample:
    print "WARNING: Just computing on a small sample"
    subsample_length = 25000


def pca_randomized(X_in, num_pca_dims):

    # take subsample
    rand_exs = np.sort(np.random.choice(
        X_in.shape[0],
        np.minimum(subsample_length, X_in.shape[0]),
        replace=False))
    X = X_in.take(rand_exs, 0)

    pca = RandomizedPCA(n_components=num_pca_dims)
    pca.fit(X)
    return pca


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


pca_savepath = paths.RenderedData.voxlets_dictionary_path + 'pca.pkl'
kmeans_savepath = paths.RenderedData.voxlets_dictionary_path + 'kmean.pkl'

# save path (open here so if an error is thrown I can catch it early...)

# initialise lists
shoeboxes = []

for count, sequence in enumerate(paths.RenderedData.train_sequence()):

    print "Processing " + sequence['name']

    # loading the data
    loadpath = paths.RenderedData.voxlets_dict_data_path + \
        sequence['name'] + '.mat'
    print "Loading from " + loadpath

    D = scipy.io.loadmat(loadpath)
    shoeboxes.append(D['shoeboxes'].astype(np.float16))

    if count > 8 and parameters.small_sample:
        print "SMALL SAMPLE: Stopping"
        break

np_all_sboxes = np.concatenate(shoeboxes, axis=0)
print "All sboxes shape is " + str(np_all_sboxes.shape)

# clustering the sboxes - but only a subsample of them for speed!
print "Doing PCA"
pca = pca_randomized(np_all_sboxes, number_pca_dims)

print "Doing Kmeans"
km = cluster_data(np_all_sboxes, subsample_length, number_clusters)

try:
    print "Saving to " + pca_savepath
    with open(pca_savepath, 'wb') as f:
        pickle.dump(pca, f)

    print "Saving to " + kmeans_savepath
    with open(kmeans_savepath, 'wb') as f:
        pickle.dump(km, f)

except:
    import pdb; pdb.set_trace()
