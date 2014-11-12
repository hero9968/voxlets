'''
Loads in the combined sboxes and clusters them to form a dictionary
smaller
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

# load in the pca kmeans
km_pca = pickle.load(open(paths.voxlet_pca_dict_path, 'rb'))
km_standard = pickle.load(open(paths.voxlet_dict_path, 'rb'))

# load in the pca components
pca = pickle.load(open(paths.voxlet_pca_path, 'rb'))


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

    # convert to pca representation
    pca_representation = pca.transform(shoebox.V.flatten())
    pca_kmeans_idx = km_pca.predict(pca_representation.flatten())
    kmeans_idx = km_standard.predict(shoebox.V.flatten())
    
    all_representations = dict(pca_representation=pca_representation, 
                                pca_kmeans_idx=pca_kmeans_idx, 
                                kmeans_idx=kmeans_idx)

    return all_representations



import multiprocessing
import functools

# parameters

small_sample = paths.small_sample
if paths.small_sample:
    number_points_from_each_image = 1
    pool = multiprocessing.Pool(4)
else:
    number_points_from_each_image = 100
    pool = multiprocessing.Pool(6)

if small_sample:
    print "WARNING: Just computing on a small sample"

for count, modelname in enumerate(paths.modelnames):

    # initialise lists
    shoeboxes = []
    all_features = []

    print "Processing " + modelname
    
    savepath = paths.bigbird_training_data_fitted_mat % modelname

    if os.path.exists(savepath):
        print "Skipping " + modelname
        continue

    vgrid = voxel_data.BigBirdVoxels()
    vgrid.load_bigbird(modelname)
    
    for view in paths.views[:45]:

        print '.'
        im = images.CroppedRGBD()
        im.load_bigbird_from_mat(modelname, view)
    
        "Sampling from image"
        idxs = random_sample_from_mask(im.mask, number_points_from_each_image)
        
        "Extracting features"
        all_features.append(im.get_features(idxs))
        
        "Now try to make this nice and like parrallel or something like what say what?"
        #these_shoeboxes = [pool_helper(idx, im, vgrid) for idx in idxs]
        these_shoeboxes = pool.map(functools.partial(pool_helper, im=im, vgrid=vgrid), idxs)
        shoeboxes.extend(these_shoeboxes)

    # perhaps *HERE* save the data for this model
    np_features = np.array(all_features)
    np_features = np_features.reshape((-1, np_features.shape[2])) # collapse 1st two dimensions

    # convert the shoeboxes to individual components
    np_pca_representation = np.array([sbox['pca_representation'] for sbox in shoeboxes])
    np_pca_representation = np_pca_representation.reshape((-1, np_pca_representation.shape[2]))
    np_kmeans_idx = np.array([sbox['kmeans_idx'] for sbox in shoeboxes]).flatten()
    np_pca_kmeans_idx = np.array([sbox['pca_kmeans_idx'] for sbox in shoeboxes]).flatten()

    print "PCA is shape " + str(np_pca_representation.shape)
    print "kmeans is shape " + str(np_kmeans_idx.shape)
    print "pca kmeans is shape " + str(np_pca_kmeans_idx.shape)

    print "Features are shape " + str(np_features.shape)
    D = dict(pca_representation=np_pca_representation, 
                pca_kmeans_idx=np_pca_kmeans_idx,
                kmeans_idx=np_kmeans_idx, 
                features=np_features)

    scipy.io.savemat(savepath, D, do_compression=True)

    if count > 4 and small_sample:
        print "Ending now"
        break
