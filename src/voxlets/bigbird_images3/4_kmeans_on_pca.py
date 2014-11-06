import numpy as np
import cPickle as pickle
import scipy.io
import sys, os
sys.path.append(os.path.expanduser('~/projects/shape_sharing/src/'))
#from sklearn.cluster import MiniBatchKMeans
from sklearn.cluster import KMeans

from common import paths
from common import voxel_data
from common import images

# parameters
subsample_length = 15000 # how many points to cluster with
number_clusters = 200
small_sample = paths.small_sample
if small_sample: 
    print "WARNING: Just computing on a small sample"
    subsample_length = 1000

# load in the pca data
f = open(paths.voxlet_pca_path, 'rb')
pca = pickle.load(f)
f.close()

def cluster_data(X, local_subsample_length, num_clusters):

    # take subsample
    if local_subsample_length > X.shape[0]:
        print "Too long! Making smaller"
        X_subset = X
    else:
        to_use_for_clustering = np.random.randint(0, X.shape[0], size=(local_subsample_length))
        X_subset = X[to_use_for_clustering, :]

    print X.shape
    print X_subset.shape

    # doing clustering
    km = KMeans(n_clusters=num_clusters)
    km.fit(X_subset)
    return km

# initialise lists
shoeboxes = []

for count, modelname in enumerate(paths.train_names):

    print "Processing " + modelname

    # loading the data
    loadpath = paths.bigbird_training_data_mat % modelname
    print "Loading from " + loadpath
    D = scipy.io.loadmat(loadpath)
    temp = D['shoeboxes'].astype(np.float16)
    shoeboxes.append(temp)


    if count > 3 and small_sample:
        print "SMALL SAMPLE: Stopping"
        break

# apply the pca to the shoeboxes
np_all_shoeboxes = np.concatenate(shoeboxes, axis=0)
pca_shoeboxes = pca.transform(np_all_shoeboxes)

print "pca_shoeboxes shape is " + str(pca_shoeboxes.shape)
print "pca_shoeboxes shape is " + str(np_all_shoeboxes.shape)

# clustering the sboxes - but only a subsample of them for speed!
print "Doing clustering"
km = cluster_data(pca_shoeboxes, subsample_length, number_clusters)

print "Saving to " + paths.voxlet_pca_dict_path
pickle.dump(km, open(paths.voxlet_pca_dict_path, 'wb'))
