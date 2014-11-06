import numpy as np
import cPickle as pickle
import scipy.io
import sys, os
sys.path.append(os.path.expanduser('~/projects/shape_sharing/src/'))

from sklearn.decomposition import RandomizedPCA

from common import paths
from common import voxel_data
from common import images


# parameters
subsample_length = 2500 # how many points to cluster with
number_pca_dims = 50
small_sample = paths.small_sample
if small_sample: 
    print "WARNING: Just computing on a small sample"
    subsample_length = 1000

# save path
f = open(paths.voxlet_pca_path, 'wb')

def pca_1(X_in, num_pca_dims):

    rand_exs = np.sort(np.random.choice(X_in.shape[0], np.minimum(subsample_length, X_in.shape[0]), replace=False))
    X = X_in.take(rand_exs, 0)

    X = X - np.tile(np.mean(X, 0), (X.shape[0], 1))
    (l, M) = np.linalg.eig(np.dot(X.T, X))
    X_ds = np.dot(X, M[:, 0:num_pca_dims])
    return X_ds


def pca_randomized(X_in, num_pca_dims):

    # take subsample
    rand_exs = np.sort(np.random.choice(X_in.shape[0], np.minimum(subsample_length, X_in.shape[0]), replace=False))
    X = X_in.take(rand_exs, 0)

    pca = RandomizedPCA(n_components=num_pca_dims)
    pca.fit(X)
    return pca


# initialise lists
shoeboxes = []

for count, modelname in enumerate(paths.train_names):

    print "Processing " + modelname

    # loading the data
    loadpath = paths.bigbird_training_data_mat % modelname
    print "Loading from " + loadpath

    D = scipy.io.loadmat(loadpath)
    shoeboxes.append(D['shoeboxes'].astype(np.float16))
    
    if count > 2 and small_sample:
        print "SMALL SAMPLE: Stopping"
        break

np_all_sboxes = np.concatenate(shoeboxes, axis=0)
print "All sboxes shape is " + str(np_all_sboxes.shape)

# clustering the sboxes - but only a subsample of them for speed!
print "Doing PCA"
pca = pca_randomized(np_all_sboxes, number_pca_dims)

try:
    print "Saving to " + paths.voxlet_pca_path
    pickle.dump(pca, f)
    f.close()
except:
    import pdb; pdb.set_trace()
