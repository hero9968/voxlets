import numpy as np
import cPickle as pickle
import scipy.io
import sys
import os
sys.path.append(os.path.expanduser('~/projects/shape_sharing/src/'))

from sklearn.decomposition import RandomizedPCA
from sklearn.cluster import MiniBatchKMeans
# from sklearn.manifold import LocallyLinearEmbedding

import real_data_paths as paths
import real_params as parameters

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

# initialise lists
shoeboxes = []
features = []

for count, sequence in enumerate(paths.train_data):

    # print "Processing " + sequence['name']

    # loading the data
    loadpath = paths.voxlets_dict_data_path + \
        sequence['name'] + '.pkl'
    print "Loading from " + loadpath

    try:
        with open(loadpath, 'r') as f:
            D = pickle.load(f)
        print D['shoeboxes'].shape

    except:
        print "failed"
        continue

    shoeboxes.append(D['shoeboxes'].reshape(50, -1).astype(np.float16))
    print D['shoeboxes'].shape
    # features.append(D['cobweb'].astype(np.float16))

    if count > parameters.max_sequences:
        print "SMALL SAMPLE: Stopping"
        break

np_all_sboxes = np.concatenate(shoeboxes, axis=0).astype(np.float16)
# np_all_features = np.concatenate(features, axis=0).astype(np.float16)
shoeboxes = features = None
print "All sboxes shape is " + str(np_all_sboxes.shape)
# print "Features shape is " + str(np_all_features.shape)
# print np_all_features.dtype
print np_all_sboxes.dtype

print "There are %d nans in sboxes" % np.isnan(np_all_sboxes).sum()

# Now removing the nans from the shoeboxes and making the masks separate...
np_all_masks = np.isnan(np_all_sboxes).astype(np.float16)
np_all_sboxes[np_all_masks == 1] = np.nanmax(np_all_sboxes)

print "There are %d nans in sboxes" % np.isnan(np_all_sboxes).sum()

# Replacing nans with a low number in the features, hopefully will work...
# np_all_features[np.isnan(np_all_features)] = -parameters.mu

for name, np_array in zip(
        ('shoeboxes', 'masks'),
        (np_all_sboxes, np_all_masks)):

    # clustering the sboxes - but only a subsample of them for speed!
    print "Doing PCA"
    pca = pca_randomized(
        np_array,
        parameters.VoxletTraining.pca_subsample_length,
        parameters.VoxletTraining.number_pca_dims)

    try:
        pca_savepath = paths.voxlets_dictionary_path + name + \
            '_pca.pkl'

        print "Saving to " + pca_savepath
        with open(pca_savepath, 'wb') as f:
            pickle.dump(pca, f, pickle.HIGHEST_PROTOCOL)

    except:
        import pdb; pdb.set_trace()
