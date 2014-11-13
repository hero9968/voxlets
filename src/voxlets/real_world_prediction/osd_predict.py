'''
predicting for the osd dataset
'''


import numpy as np
import matplotlib.pyplot as plt 
import cPickle as pickle
import sys, os
sys.path.append(os.path.expanduser('~/projects/shape_sharing/src/'))
sys.path.append('../bigbird_images3/')

from common import paths
from common import voxel_data
from common import mesh
from common import images
from common import features
from thickness_regress import combine_data
import reconstructer

############################################################
print "Setting parameters"
max_points = 200
print "WARNING - only doing with 200 points"
number_samples = 2000
padding_value = 0.15 # in future pass this in
savefolder = paths.base_path + "other_3D/osd/OSD-0.2-depth/predictions/"

############################################################
print "Loading forest"
oma_forest = pickle.load(open(paths.voxlet_model_oma_path, 'rb'))

############################################################
print "Main loop"
f = open('./names.txt', 'r')
for fline in f:
    name = fline.strip()

    savepath = savefolder + name + ".pkl"

    print "Loading image " + name
    im = images.RealRGBD()
    im.load_from_mat(name)
    im.disp_channels()

    print "Reconstructing with oma forest"
    rec = reconstructer.Reconstructer(reconstruction_type='kmeans_on_pca', combine_type='modal_vote')
    rec.set_forest(oma_forest)
    rec.set_test_im(im)
    rec.sample_points(number_samples)
    rec.initialise_output_grid(method='from_image')
    accum = rec.fill_in_output_grid_oma(max_points=max_points)
    prediction = accum.compute_average(nan_value=0.03)

    print "Force the base to be solid"
    padding_value = 0.15 # in future pass this in
    base_height = padding_value / accum.vox_size
    accum.V[:, :, :base_height] = 0.03

    print "Saving result to " + savepath
    pickle.dump(accum, open(savepath, 'wb'))

    print "Breaking"
    break

print "Done"
