'''
extracts training data from bigbird for the implicit voxel prediction method
'''

import numpy as np
import scipy.io
import sys, os
sys.path.append(os.path.expanduser('~/projects/shape_sharing/src/'))

import multiprocessing
import functools

from common import paths
from common import voxel_data
from common import images
from features import line_casting

#########################################
save_path_template = paths.base_path + '/implicit/bigbird/training_data_rotated/%s_%s.mat'

multicore = True

#########################################
def pool_helper(view_idx, modelname, gt_grid):

    save_name = save_path_template % (modelname, view_idx)
    if os.path.exists(save_name):
        print "Skipping " + modelname + " " + view_idx
        return

    # now load the image
    im = images.CroppedRGBD()
    im.load_bigbird_from_mat(modelname, view_idx)

    # create a padded version of the gt grid
    expanded_gt = voxel_data.expanded_grid(gt_grid)
    expanded_gt.fill_from_grid(gt_grid)

    # getting the known full and empty voxels based on the depth image
    known_full_voxels = voxel_data.get_known_full_grid(im, expanded_gt)
    known_empty_voxels = voxel_data.get_known_empty_grid(im, expanded_gt, known_full_voxels)[0]

    # now use final_V to extract the 3D features
    gt_tsdf_V = expanded_gt.compute_tsdf(0.03)

    X, Y = line_casting.feature_pairs_3d(known_empty_voxels, known_full_voxels, gt_tsdf_V,
                                         samples=200, base_height=15, autorotate=True)

    print "Done view " + view_idx + " with shapes " + str(X.shape) + " " + str(Y.shape)
    print "Saving to " + save_name
    D = dict(X=X, Y=Y)
    scipy.io.savemat(save_name, D, do_compression=True)
    return 


#########################################
if __name__ == '__main__':

    if multicore:
        pool = multiprocessing.Pool(multiprocessing.cpu_count())

    for modelname in paths.modelnames:

        # load in the ground truth voxel grid 
        gt_grid = voxel_data.BigBirdVoxels()
        gt_grid.load_bigbird(modelname)

        # multicore version uses partial to deal with multiple input args to pool_helper
        if multicore:
            pool.map(functools.partial(pool_helper, modelname=modelname, gt_grid=gt_grid), paths.views[:45])

        else:
            for view_idx in paths.views[:45]:
                pool_helper(view_idx, modelname, gt_grid)

        print "Done model " + modelname