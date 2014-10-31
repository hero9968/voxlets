'''
loads all the compted shoebox voxlets and clusters them using kmeans or something
to make a dictionary
this is then saved to disk
'''

import numpy as np
import sys
import cPickle as pickle

sys.path.append('/Users/Michael/projects/shape_sharing/src/structured_train/')
from thickness import paths

# setting parameters
number_clusters = 100
shoebox_kmeans_path = paths.base_path + "voxlets/shoebox_dictionary_training.pkl"

# loading the extracted shoeboxes
print "Loading shoeboxes..."
shoebox_path = paths.base_path + "voxlets/all_shoeboxes.pkl"
D = pickle.load(open(shoebox_path, 'rb'))

# combining all the shoeboxes together
print "Combining..."
all_shoeboxes = []
for modelname in paths.train_names:
    model_shoeboxes = D[modelname]
    all_v = [shoebox.V for shoebox in model_shoeboxes]
    all_shoeboxes.append(np.array(all_v).astype(np.int))

num_voxels = np.prod(all_v[0].shape)
all_shoeboxes = np.array(all_shoeboxes).reshape((-1, num_voxels))
print all_shoeboxes.shape

# clustering...
print "Doing clustering..."
from sklearn.cluster import MiniBatchKMeans
km = MiniBatchKMeans(n_clusters=150)
km.fit(all_shoeboxes.astype(np.float16))

# saving the km object
print "Saving..."
pickle.dump(km, open(shoebox_kmeans_path, 'wb'))