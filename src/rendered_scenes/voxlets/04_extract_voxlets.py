'''
Loads in the combined sboxes and clusters them to form a dictionary
smaller
'''
import numpy as np
import cPickle as pickle
import sys
import os
import yaml
from time import time
import scipy.io

sys.path.append('/Users/Michael/projects/shape_sharing/src/')
from common import paths
from common import voxel_data
from common import images
from common import parameters
from common import features

# parameters
number_points_from_each_image = 10
multiproc = False


def pool_helper(index, im, vgrid):

    world_xyz = im.get_world_xyz()
    world_norms = im.get_world_normals()

    # convert to linear idx
    point_idx = index[0] * im.mask.shape[1] + index[1]

    shoebox = voxel_data.ShoeBox(parameters.Voxlet.shape, np.float32)  # grid size
    shoebox.set_p_from_grid_origin(parameters.Voxlet.centre)  # m
    shoebox.set_voxel_size(parameters.Voxlet.size)  # m
    shoebox.initialise_from_point_and_normal(
        world_xyz[point_idx], world_norms[point_idx], np.array([0, 0, 1]))

    # convert the indices to world xyz space
    shoebox.fill_from_grid(vgrid)
    print "Sum is %f" % np.sum(shoebox.V.flatten())
    return shoebox.V.flatten()

# need to import these *after* the pool helper has been defined
if multiproc:
    import multiprocessing
    import functools
    pool = multiprocessing.Pool(parameters.cores)

for count, sequence in enumerate(paths.RenderedData.train_sequence()):

    print "Processing " + sequence['name']

    # load in the ground truth grid for this scene, and converting nans
    scene_folder = paths.scenes_location + sequence['scene']
    gt_vox = voxel_data.load_voxels(scene_folder + '/voxelgrid.pkl')
    gt_vox.V[np.isnan(gt_vox.V)] = -0.1
    gt_vox.set_origin(gt_vox.origin)

    # loading this frame
    frame_data = paths.RenderedData.load_scene_data(
        sequence['scene'], sequence['frames'][0])
    im = images.RGBDImage.load_from_dict(
        paths.scenes_location + sequence['scene'], frame_data)

    # computing normals...
    norm_engine = features.Normals()
    im.normals = norm_engine.compute_normals(im)

    "Sampling from image"
    idxs = im.random_sample_from_mask(number_points_from_each_image)

    "Extracting features"
    ce = features.CobwebEngine(t=5, fixed_patch_size=False)
    ce.set_image(im)
    np_features = np.array(ce.extract_patches(idxs))

    "Now try to make this nice and like parrallel or something...?"
    t1 = time()
    if multiproc:
        shoeboxes = pool.map(
            functools.partial(pool_helper, im=im, vgrid=gt_vox), idxs)
    else:
        shoeboxes = [pool_helper(idx, im=im, vgrid=gt_vox) for idx in idxs]
    print "Took %f s" % (time() - t1)

    np_sboxes = np.array(shoeboxes)

    print "Shoeboxes are shape " + str(np_sboxes.shape)
    print "Features are shape " + str(np_features.shape)

    D = dict(shoeboxes=np_sboxes, features=np_features)
    savepath = paths.RenderedData.voxlets_dict_data_path + \
        sequence['name'] + '.mat'
    print savepath
    scipy.io.savemat(savepath, D, do_compression=True)

    if count > 4 and paths.small_sample:
        print "Ending now"
        break
