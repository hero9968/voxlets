'''
Loads in the combined sboxes and clusters them to form a dictionary
'''
import numpy as np
import sys, os
import scipy.io
import cPickle as pickle

sys.path.append(os.path.expanduser('~/projects/shape_sharing/src/'))
from common import paths

number_clusters = 200

print "Loading shoeboxes..."
all_training_sbox_path = paths.base_path + "voxlets/dict/training_sboxes_from_images.npy"
all_sboxes = np.load(all_training_sbox_path).reshape((-1, 16000))
print all_sboxes.shape
# clustering...
print "Doing clustering..."
from sklearn.cluster import MiniBatchKMeans
km = MiniBatchKMeans(n_clusters=number_clusters)
km.fit(all_sboxes.astype(np.float16))

# saving the km object
print "Saving..."
shoebox_kmeans_path = paths.base_path + "voxlets/dict/clusters_from_images.pkl"
pickle.dump(km, open(shoebox_kmeans_path, 'wb'))

print "Converting to voxlet list"
voxlet_list = []
for centre in km.cluster_centers_:
    voxlet_list.append(centre.reshape((20, 40, 20)))
D = {}
D['voxlet_list'] = voxlet_list
D['voxel_size'] = 0.005
D['normal_direction'] = [0, 1, 0]
D['reference_point_location_in_idx'] = [10, 10, 10]

shoebox_kmeans_path2 = paths.base_path + "voxlets/dict/clusters_from_images_as_list.pkl"
pickle.dump(km, open(shoebox_kmeans_path2, 'wb'))
