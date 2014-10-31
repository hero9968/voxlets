'''computes a shoebox for many points from bigbird images'''
import numpy as np
import cPickle as pickle
import sys, os
sys.path.append(os.path.expanduser('~/projects/shape_sharing/src/structured_train/'))
import scipy.io

from thickness import voxel_data
from thickness import paths
from thickness import mesh
from thickness import images
from thickness import features


'''PARAMETERS'''
overwrite = False

# params for the shoeboxes
shoebox_gridsize = (20, 40, 20)
shoebox_p_from_origin = np.array((0.05, 0.05, 0.05))
shoebox_voxelsize = 0.2/40.0

number_points_from_each_image = 100


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


for modelname in paths.modelnames:

    # creating folder if it doesn't exist
    savefolder = paths.base_path + "voxlets/bigbird/%s/" % modelname
    if not os.path.exists(savefolder):
        os.mkdir(savefolder)

    # skipping this model if it seems we've already saved all the required files       
    if num_files_in_dir(savefolder) == 75 and not overwrite:
        print "Skipping model " + modelname
        continue

    # loading the voxel grid for this model
    print "Loading voxel data for " + modelname
    vgrid = voxel_data.BigBirdVoxels()
    vgrid.load_bigbird(modelname)

    # loop over the different images of this object:
    for view in paths.views:

        savepath = savefolder + "%s.mat" % view

        # see if should skip
        if os.path.exists(savepath) and not overwrite:
            print "Skipping " + savepath
            continue

        # load the image and the camera
        im = images.CroppedRGBD()
        try:
            im.load_bigbird_from_mat(modelname, view)
        except:
            print "Could not load bigbird!"
            print "Skipping " + modelname + " " + view_idx
            continue

        # sample points from the image mask
        mask = ~np.isnan(im.frontrender)
        sampled_points = random_sample_from_mask(mask, number_points_from_each_image)

        # now compute the shoeboxes
        im_shoeboxes = shoeboxes_from_image(im, vgrid, sampled_points)

        # now compute the features
        try:
            cobweb, spider = features_from_image(im, sampled_points)
        except:
            print "Could not get features!"
            print "Skipping " + modelname + " " + view_idx
            continue

        # save these to a file
        D = dict(sboxes=im_shoeboxes, cobweb=cobweb, spider=spider, sampled_points=sampled_points)
        scipy.io.savemat(savepath, D, do_compression=True)

        print "Done " + savepath

    print "Done model " + modelname

