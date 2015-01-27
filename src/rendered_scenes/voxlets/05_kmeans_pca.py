import numpy as np
import cPickle as pickle
import scipy.io
import sys
import os
sys.path.append(os.path.expanduser('~/projects/shape_sharing/src/'))

from sklearn.decomposition import RandomizedPCA

from common import paths
from common import parameters
from common import voxel_data
from common import images

# parameters
subsample_length = 2500

# how many points to cluster with
number_pca_dims = 50
if parameters.small_sample:
    print "WARNING: Just computing on a small sample"
    subsample_length = 1000


def pca_randomized(X_in, num_pca_dims):

    # take subsample
    rand_exs = np.sort(np.random.choice(X_in.shape[0], np.minimum(subsample_length, X_in.shape[0]), replace=False))
    X = X_in.take(rand_exs, 0)

    pca = RandomizedPCA(n_components=num_pca_dims)
    pca.fit(X)
    return pca

pca_savepath = paths.RenderedData.voxlets_dictionary_path + 'pca.pkl'

# save path (open here so if an error is thrown I can catch it early...)
with open(pca_savepath, 'wb') as f:

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

        if count > 2 and parameters.small_sample:
            print "SMALL SAMPLE: Stopping"
            break

    np_all_sboxes = np.concatenate(shoeboxes, axis=0)
    print "All sboxes shape is " + str(np_all_sboxes.shape)

    # clustering the sboxes - but only a subsample of them for speed!
    print "Doing PCA"
    pca = pca_randomized(np_all_sboxes, number_pca_dims)

    try:
        print "Saving to " + pca_savepath
        pickle.dump(pca, f)
    except:
        import pdb; pdb.set_trace()
