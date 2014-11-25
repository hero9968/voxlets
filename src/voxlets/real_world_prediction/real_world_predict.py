'''
predicting for the osd dataset
'''


import numpy as np
import matplotlib.pyplot as plt 
import cPickle as pickle
import sys, os
sys.path.append(os.path.expanduser('~/projects/shape_sharing/src/'))
sys.path.append('../bigbird_images3/')
import scipy.io

from common import paths
from common import voxel_data
from common import mesh
from common import images
from common import features
from thickness_regress import combine_data
import reconstructer

############################################################
print "Setting parameters"
max_points = 500
print "WARNING - only doing with 200 points"
number_samples = 500
padding_value = 0.15 # in future pass this in
savefolder = paths.base_path + "voxlets/from_biryani/predictions/"

############################################################
print "Loading forest"
oma_forest = pickle.load(open(paths.voxlet_model_oma_path, 'rb'))

############################################################
print "Main loop"
'''names = {'frame_20141120T134535.682741_', 'frame_20141120T185954.959422_'}
'''
names = ['frame_20141121T145136.414620_']

'''frame_20141120T215459.461384_']
'''
'''frame_20141120T134535.682741_', 
'frame_20141120T185954.959422_', 
'frame_20141120T213448.378796_', 
'frame_20141120T213442.525035_', 
'frame_20141120T213440.257049_', 
'frame_20141120T213449.827419_']
'''
for name in names:

    savepath = savefolder + name + ".mat"
    savepathpkl = savefolder + name + ".pkl"

    print "Loading image " + name
    im = images.RealRGBD()
    im.load_from_mat(name)
    '''im.disp_channels()
    '''
    print "Reconstructing with oma forest"
    rec = reconstructer.Reconstructer(reconstruction_type='kmeans_on_pca', combine_type='modal_vote')
    rec.set_forest(oma_forest)
    rec.set_test_im(im)
    rec.sample_points(number_samples)
    rec.initialise_output_grid(method='from_image')
    accum = rec.fill_in_output_grid_oma(max_points=max_points)
    prediction = accum.compute_average(nan_value=0.03)

    print "Force the base to be solid"
    #padding_value = 0.15 # in future pass this in
    #base_height = padding_value / accum.vox_size
    #accum.V[:, :, :base_height] = 0.03

    print "Saving result to " + savepath
    pickle.dump(accum, open(savepathpkl, 'wb'))
    
    D = dict(prediction=accum.V)
    scipy.io.savemat(savepath, D)
    
    #print "Breaking"
    #break

print "Done"

