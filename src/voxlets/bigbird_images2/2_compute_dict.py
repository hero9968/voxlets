
import numpy as np
import cPickle as pickle
import scipy.io
import sys
sys.path.append('/Users/Michael/projects/shape_sharing/src/')
from sklearn.cluster import MiniBatchKMeans

from common import paths
from common import voxel_data
from common import images

# parameters
subsample_length = 10000 # how many points to cluster with
number_clusters = 200
small_sample = False
if small_sample: print "WARNING: Just computing on a small sample"

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
    km = MiniBatchKMeans(n_clusters=num_clusters)
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
    shoeboxes.append(D['shoeboxes'])
    
    if count > 2 and small_sample:
        print "SMALL SAMPLE: Stopping"
        break

for s in shoeboxes:
    print s.shape
np_all_sboxes = np.concatenate(shoeboxes, axis=0)
print "All sboxes shape is " + str(np_all_sboxes.shape)

# clustering the sboxes - but only a subsample of them for speed!
print "Doing clustering"
km = cluster_data(np_all_sboxes, subsample_length, number_clusters)

print "Saving to " + paths.voxlet_dict_path
pickle.dump(km, open(paths.voxlet_dict_path, 'wb'))
