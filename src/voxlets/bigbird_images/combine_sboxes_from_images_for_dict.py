'''
loads all the compted shoebox voxlets and clusters them using kmeans or something
to make a dictionary
this is then saved to disk
actually, this had better be some kind of combine data script
'''

import numpy as np
import sys, os
import scipy.io
import cPickle as pickle

sys.path.append(os.path.expanduser('~/projects/shape_sharing/src/'))
sys.path.append('..')
from common import paths

# setting parameters
num_to_sample_from_each_model = 1000
shoebox_kmeans_path = paths.base_path + "voxlets/shoebox_dictionary_training_images.pkl"

from shoebox_helpers import *

all_sboxes = []

for modelname in paths.train_names:
    this_model_sbox_list = []
    for view in paths.views:
        
        try:
            these_V = load_bigbird_shoeboxes(modelname, view)            
            this_model_sbox_list.append(these_V)
            print "Done " + view
        except:
            print "Failed to do model %s and view %s " % (modelname, view)

    # finding out how many voxel in each sbox
    points_in_sbox = np.array(this_model_sbox_list).shape[2]

    this_model_sboxes = np.array(this_model_sbox_list).reshape((-1, points_in_sbox))
    print "Before sampling: " + str(this_model_sboxes.shape)

    # perhaps here subsample this model sboxes...
    random_row_idxs = np.random.choice(this_model_sboxes.shape[0], num_to_sample_from_each_model, replace=False)
    this_model_sboxes = this_model_sboxes[random_row_idxs, :]
    print "After sampling: " + str(this_model_sboxes.shape)

    all_sboxes.append(this_model_sboxes)

    print "Done model " + modelname

print np.array(all_sboxes).shape
print np.array(all_sboxes).reshape((-1, points_in_sbox)).shape

# now save all of them to disk!
all_training_sbox_path = paths.base_path + "voxlets/dict/training_sboxes_from_images"
np.save(all_training_sbox_path, all_sboxes)

