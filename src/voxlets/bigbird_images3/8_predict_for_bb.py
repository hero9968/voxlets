'''
make predictions for all of bigbird dataset
using my algorhtm
saves each prediction to disk
'''

import numpy as np
import matplotlib.pyplot as plt 
import cPickle as pickle
import sys
sys.path.append('/Users/Michael/projects/shape_sharing/src/')

from common import paths
from common import voxel_data
from common import mesh
from common import images
from common import features
import reconstructer

"Parameters"
max_points = 100

"Loading clusters and forest"
forest = pickle.load(open(paths.voxlet_model_path, 'rb'))
forest_pca = pickle.load(open(paths.voxlet_model_pca_path, 'rb'))

km = pickle.load(open(paths.voxlet_dict_path, 'rb'))
km_pca = pickle.load(open(paths.voxlet_pca_dict_path, 'rb'))

"Loading in test data"
test_view = paths.views[10]
modelname = paths.test_names[1]

vgrid = voxel_data.BigBirdVoxels()
vgrid.load_bigbird(modelname)

test_im = images.CroppedRGBD()
test_im.load_bigbird_from_mat(modelname, test_view)

"Filling the accumulator"
rec = reconstructer.Reconstructer()
rec.set_forest(forest)
rec.set_km_dict(km)
rec.initialise_output_grid(method='from_grid', gt_grid=vgrid)
rec.set_test_im(test_im)
rec.sample_points(2000)
accum1 = rec.fill_in_output_grid(max_points=max_points)

"Saving result to disk"