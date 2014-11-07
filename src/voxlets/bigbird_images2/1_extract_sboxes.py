'''
Loads in the combined sboxes and clusters them to form a dictionary
'''
import numpy as np
import matplotlib.pyplot as plt 
import cPickle as pickle
import sys
sys.path.append('/Users/Michael/projects/shape_sharing/src/')
sys.path.append('..')

from shoebox_helpers import *
from common import paths
from common import voxel_data
from common import images


def pool_helper(index, im, vgrid):


    "Extracting shoeboxes"
    world_xyz = im.get_world_xyz()
    world_norms = im.get_world_normals()


    # convert to linear idx
    point_idx = index[0] * im.mask.shape[1] + index[1]
        
    shoebox = voxel_data.ShoeBox(paths.voxlet_shape) # grid size
    shoebox.set_p_from_grid_origin(np.array(paths.voxlet_centre)) #m
    shoebox.set_voxel_size(paths.voxlet_size) #m
    shoebox.initialise_from_point_and_normal(world_xyz[point_idx], 
                                             world_norms[point_idx], 
                                             np.array([0, 0, 1]))
    
    # convert the indices to world xyz space
    shoebox_xyz_in_world = shoebox.world_meshgrid()
    shoebox_xyx_in_world_idx, valid = vgrid.world_to_idx(shoebox_xyz_in_world, True)

    sbox_idxs = shoebox_xyx_in_world_idx[valid, :]
    occupied_values = vgrid.extract_from_indices(sbox_idxs)
    shoebox.set_indicated_voxels(valid, occupied_values)
    shoebox.convert_to_tsdf(0.03)

    shoebox._clear_cache() # important for memory purposes!
    
    return shoebox.V.flatten()
    #shoeboxes.append()



import multiprocessing
pool = multiprocessing.Pool(10)
import functools


# parameters
number_points_from_each_image = 100
if paths.host_name != 'troll':
    small_sample = True
else:
    small_sample = False

if small_sample:
    print "WARNING: Just computing on a small sample"


for count, modelname in enumerate(paths.modelnames):

    # initialise lists
    shoeboxes = []
    all_features = []

    print "Processing " + modelname
    
    savepath = paths.bigbird_training_data_mat_tsdf % modelname

    if os.path.exists(savepath):
        print "Skipping " + modelname
        continue

    vgrid = voxel_data.BigBirdVoxels()
    vgrid.load_bigbird(modelname)

    
    for view in paths.views:

        print '.'
        im = images.CroppedRGBD()
        im.load_bigbird_from_mat(modelname, view)
    
        "Sampling from image"
        mask = ~np.isnan(im.frontrender)
        idxs = random_sample_from_mask(mask, number_points_from_each_image)
        
        "Extracting features"
        all_features.append(im.get_features(idxs))
        
        "Now try to make this nice and like parrallel or something like what say what?"
        temp_shoeboxes = pool.map(functools.partial(pool_helper, im=im, vgrid=vgrid), idxs)
        shoeboxes.append(temp_shoeboxes)

    # perhaps *HERE* save the data for this model
    np_sboxes = np.array(shoeboxes)
    np_features = np.array(all_features)
    np_features = np_features.reshape((-1, np_features.shape[2])) # collapse 1st two dimensions

    print "Shoeboxes are shape " + str(np_sboxes.shape)
    print "Features are shape " + str(np_features.shape)
    D = dict(shoeboxes=np_sboxes, features=np_features)
    scipy.io.savemat(savepath, D, do_compression=True)



