'''
Extracts all the shoeboxes from all the training images
'''
import numpy as np
import cPickle as pickle
import sys
import os
from time import time
import scipy.io
import functools

sys.path.append(os.path.expanduser('~/projects/shape_sharing/src/'))
from common import paths
from common import voxel_data
from common import images
from common import parameters
from common import features
from common import carving

pca_savepath = paths.RenderedData.voxlets_dictionary_path + 'shoeboxes_pca.pkl'
with open(pca_savepath, 'rb') as f:
    pca = pickle.load(f)

if not os.path.exists(paths.RenderedData.voxlets_data_path):
    os.makedirs(paths.RenderedData.voxlets_data_path)


def pool_helper(index, im, vgrid, post_transform=None):
    '''
    Helper function to extract shoeboxes from specified locations in voxel
    grid.
    post_transform is a function which is applied to each shoebox
    after extraction
    '''

    world_xyz = im.get_world_xyz()
    world_norms = im.get_world_normals()

    # convert to linear idx
    point_idx = index[0] * im.mask.shape[1] + index[1]

    shoebox = voxel_data.ShoeBox(parameters.Voxlet.shape, np.float32)
    shoebox.V *= 0
    shoebox.V -= parameters.RenderedVoxelGrid.mu  # set the outside area to mu
    shoebox.set_p_from_grid_origin(parameters.Voxlet.centre)  # m
    shoebox.set_voxel_size(parameters.Voxlet.size)  # m
    shoebox.initialise_from_point_and_normal(
        world_xyz[point_idx], world_norms[point_idx], np.array([0, 0, 1]))

    # convert the indices to world xyz space
    shoebox.fill_from_grid(vgrid)

    # print a dot each time the function is run, without a new line
    # sys.stdout.write('.')
    # sys.stdout.flush()

    return post_transform(shoebox.V)


def pca_flatten(X):
    """Applied to the GT shoeboxes after extraction"""
    return pca.transform(X.flatten())


def decimate(X):
    """Applied to the feature shoeboxes after extraction"""
    rate = parameters.VoxletTraining.decimation_rate
    X_sub = X[::rate, ::rate, ::rate]
    return X_sub.flatten()


def process_sequence(sequence):

    print "\nProcessing " + sequence['name']

    print "-> Loading ground truth grid and image data"
    vox_dir = paths.RenderedData.ground_truth_voxels(sequence['scene'])
    gt_vox = voxel_data.load_voxels(vox_dir)
    gt_vox.V[np.isnan(gt_vox.V)] = -parameters.RenderedVoxelGrid.mu
    gt_vox.set_origin(gt_vox.origin)

    frame_data = paths.RenderedData.load_scene_data(
        sequence['scene'], sequence['frames'][0])
    im = images.RGBDImage.load_from_dict(
        paths.RenderedData.scene_dir(sequence['scene']), frame_data)

    print "-> Computing normals"
    norm_engine = features.Normals()
    im.normals = norm_engine.compute_normals(im)

    print "-> Sampling from image"
    idxs = im.random_sample_from_mask(
        parameters.VoxletTraining.number_points_from_each_image)

    print "-> Voxel carving"
    video = images.RGBDVideo.init_from_images([im])
    carver = carving.Fusion()
    carver.set_video(video)
    carver.set_voxel_grid(gt_vox)
    im_tsdf, visible = carver.fuse(mu=parameters.RenderedVoxelGrid.mu)
    im_tsdf.V[np.isnan(im_tsdf.V)] = -parameters.RenderedVoxelGrid.mu

    print "-> Extracting shoeboxes and features..."
    t1 = time()
    gt_shoeboxes = [pool_helper(
        idx, im=im, vgrid=gt_vox, post_transform=pca_flatten) for idx in idxs]
    view_shoeboxes = [pool_helper(
        idx, im=im, vgrid=im_tsdf, post_transform=decimate) for idx in idxs]
    print "\n-> ...Took %f s" % (time() - t1)

    np_sboxes = np.vstack(gt_shoeboxes)
    np_features = np.array(view_shoeboxes)

    print "-> ...Shoeboxes are shape " + str(np_sboxes.shape)
    print "-> ...Features are shape " + str(np_features.shape)

    savepath = paths.RenderedData.voxlets_data_path + \
        sequence['name'] + '.mat'
    print "-> Saving to " + savepath
    D = dict(shoeboxes=np_sboxes, features=np_features)
    scipy.io.savemat(savepath, D, do_compression=True)


if parameters.multicore:
    # need to import these *after* pool_helper has been defined
    import multiprocessing
    pool = multiprocessing.Pool(parameters.cores)
    mapper = pool.map
else:
    mapper = map


if __name__ == "__main__":

    tic = time()
    mapper(process_sequence, paths.RenderedData.train_sequence())
    print "In total took %f s" % (time() - tic)
