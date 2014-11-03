'''
some functions to help with shoebox extraction
'''
import numpy as np
import scipy.io
import sys, os
sys.path.append(os.path.expanduser('~/projects/shape_sharing/src/'))

from common import voxel_data
from common import mesh
from common import paths

# params for the shoeboxes
shoebox_gridsize = (20, 40, 20)
shoebox_p_from_origin = np.array((0.05, 0.05, 0.05))
shoebox_voxelsize = 0.2/40.0


def shoeboxes_from_image(im, vgrid, indices):
    '''
    given an image and a voxel grid, computes sboxes for points on the image...
    '''
    cam = mesh.Camera()
    cam.load_bigbird_matrices(im.modelname, im.view_idx)
    cam.adjust_intrinsic_scale(0.5) # as the image is half-sized

    world_xyz = cam.inv_project_points(im.get_uvd())
    world_norms = cam.inv_transform_normals(im.normals)

    image_shoeboxes = []

    for index in indices:

        # convert index to row/col notation
        row, col = index
        im_width = im.mask.shape[1]
        linear_index = row * im_width + col

        # extract 3d location for this point    
        xyz = world_xyz[linear_index]
        norm = world_norms[linear_index]
        updir = np.array([0, 0, 1])

        # create single shoebox
        shoebox = voxel_data.ShoeBox(shoebox_gridsize) # grid size
        shoebox.set_p_from_grid_origin(shoebox_p_from_origin) # metres
        shoebox.set_voxel_size(shoebox_voxelsize) # metres
        shoebox.initialise_from_point_and_normal(xyz, norm, updir)

        # convert the indices to world xyz space
        shoebox_xyz_in_world = shoebox.world_meshgrid()
        shoebox_xyx_in_world_idx, valid = vgrid.world_to_idx(shoebox_xyz_in_world, True)

        # fill in the shoebox voxels
        idxs = shoebox_xyx_in_world_idx[valid, :]
        occupied_values = vgrid.extract_from_indices(idxs)
        shoebox.set_indicated_voxels(valid, occupied_values)

        image_shoeboxes.append(shoebox)

    return image_shoeboxes



def features_from_image(im, indices):
    '''
    extract cobweb and spider features from image at specified locations
    '''
    patch_engine = features.CobwebEngine(t=5, fixed_patch_size=False)
    patch_engine.set_image(im)
    patch_features = patch_engine.extract_patches(indices)

    spider_engine = features.SpiderEngine(im)
    spider_features = spider_engine.compute_spider_features(indices)

    return patch_features, spider_features


def random_sample_from_mask(mask, num_samples):
    '''sample random points from a mask'''
    
    indices = np.array(np.nonzero(mask)).T
    samples = np.random.randint(0, indices.shape[0], num_samples)
    return indices[samples, :]


def num_files_in_dir(dirname):
    return len([name for name in os.listdir(dirname) if os.path.isfile(os.path.join(dirname, name))])



def load_bigbird_shoeboxes(modelname, view):
    loadpath = paths.base_path + "voxlets/bigbird/%s/%s.mat" % (modelname, view)

    D = scipy.io.loadmat(loadpath)

    # loading in the shoeboxes (need some magic to sort out the matlab crap)
    each_view_sbox = np.array([sbox[0][0][4] for sbox in D['sboxes'].flatten()])
    num_boxes = each_view_sbox.shape[0]
    image_sboxes = np.array(each_view_sbox).reshape((num_boxes, -1))

    return image_sboxes


def load_bigbird_shoeboxes_and_features(modelname, view):
    loadpath = paths.base_path + "voxlets/bigbird/%s/%s.mat" % (modelname, view)

    D = scipy.io.loadmat(loadpath)

    # loading in the shoeboxes (need some magic to sort out the matlab crap)
    each_view_sbox = np.array([sbox[0][0][4] for sbox in D['sboxes'].flatten()])
    num_boxes = each_view_sbox.shape[0]
    image_sboxes = np.array(each_view_sbox).reshape((num_boxes, -1))

    # now need to get features
    image_cobwebs = np.array(D['cobweb'])
    image_spider = np.array(D['spider'])

    return image_sboxes, image_cobwebs, image_spider


    