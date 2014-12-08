# Now doing this in a loop
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

import scipy.io
import os
from os import listdir
from os.path import isfile, join


for modelname in paths.test_names:
    
    # choosing if to skip
    mypath = paths.base_path + "voxlets/bigbird/predictions/visible_pixels/"
    counter = 0
    onlyfiles = [ f for f in listdir(mypath) if isfile(join(mypath,f)) ]
    print onlyfiles
    for ff in onlyfiles:
        if modelname in ff:
            counter += 1
    print "counter is " + str(counter)
    if counter >= 75:
        print "Skp folder " + modelname
        continue

    # load in the ground truth voxel grid 
    gt_grid = voxel_data.BigBirdVoxels()
    gt_grid.load_bigbird(modelname)
    
    accum = voxel_data.expanded_grid_accum(gt_grid)
    accum.fill_from_grid(gt_grid)
    
    for view_idx in paths.views:
        savepath = paths.voxlet_prediction_path % ('visible_pixels', modelname, view_idx)
        if os.path.isfile(savepath):
            print "Skp " + view_idx
            continue
            
        # now load the image
        im = images.CroppedRGBD()
        im.load_bigbird_from_mat(modelname, view_idx)
        world_xyz = im.get_world_xyz()
        
        # populate the grid
        accum.V *= 0
        idx = accum.world_to_idx(world_xyz)
        valid = accum.find_valid_idx(idx)
        accum.set_idxs(idx[valid, :], 1)
        
        # save the grid
        savepath = paths.voxlet_prediction_path % ('visible_pixels', modelname, view_idx)
        D = dict(prediction=accum.V, gt=gt_grid)
        f = open(savepath, 'wb')
        scipy.io.savemat(f, D, do_compression=True)
        f.close()

        
#        print "Done " + view_idx
        
    print "Done " + modelname