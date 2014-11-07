'''
aside script to reconstruct the real clusters from the data
sorts them by size of cluster
saves to disk
'''

import numpy as np
import matplotlib.pyplot as plt 
import cPickle as pickle
import sys
sys.path.append('/Users/Michael/projects/shape_sharing/src/')
from common import paths

'''
must make sure to reconstruct the full ones
'''

# loading in the PCA clusters
pca_comp = pickle.load(open(paths.voxlet_pca_path[:-6], 'rb'))
pca_dict = pickle.load(open(paths.voxlet_pca_dict_path[:-6], 'rb'))

order = np.argsort(np.bincount(pca_dict.labels_))[::-1]
all_voxlets = []

for count, idx in enumerate(order):

    cen = pca_dict.cluster_centers_[idx]

    vector_center = pca_comp.inverse_transform(cen)
    reconstructed_centre = vector_center.reshape(paths.voxlet_shape)

    all_voxlets.append(reconstructed_centre)

pickle.dump(all_voxlets, open('all_voxlets.pkl', 'wb'))

# to copy to the prism workspace:
# rsync -p all_voxlets.np newgate:/cs/research/vecg/prism/data1/michael/


