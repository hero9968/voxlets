'''
Extracts all the shoeboxes from all the training images
'''
import numpy as np
import cPickle as pickle
import sys
import os
from time import time
import scipy.io
import logging
logging.basicConfig(level=logging.DEBUG)

sys.path.append(os.path.expanduser('~/projects/shape_sharing/src/'))
from common import paths
from common import voxel_data
from common import images
from common import parameters
from common import features
from common import carving
from common import voxlets

pca_savepath = paths.RenderedData.voxlets_dictionary_path + 'shoeboxes_pca.pkl'
with open(pca_savepath, 'rb') as f:
    pca = pickle.load(f)

features_pca_savepath = paths.RenderedData.voxlets_dictionary_path + 'features_pca.pkl'
with open(features_pca_savepath, 'rb') as f:
    features_pca = pickle.load(f)

print "PCA components is shape ", pca.components_.shape
print "Features PCA components is shape ", features_pca.components_.shape

if not os.path.exists(paths.RenderedData.voxlets_data_path):
    os.makedirs(paths.RenderedData.voxlets_data_path)


def pca_flatten(sbox):
    """Applied to the GT shoeboxes after extraction"""
    return pca.transform(sbox.V.flatten())


def feature_transform(sbox):
    """Applied to the feature shoeboxes after extraction"""

    if parameters.VoxletTraining.feature_transform == 'pca':
        return features_pca.transform(sbox.V.flatten())

    elif parameters.VoxletTraining.feature_transform == 'decimate':
        rate = parameters.VoxletTraining.decimation_rate
        X_sub = sbox.V[::rate, ::rate, ::rate]
        return X_sub.flatten()


def process_sequence(sequence):

    logging.info("Processing " + sequence['name'])

    logging.debug("Loading ground truth grid and image data")
    vox_dir = paths.RenderedData.ground_truth_voxels(sequence['scene'])
    gt_vox = voxel_data.load_voxels(vox_dir)
    gt_vox.V[np.isnan(gt_vox.V)] = -parameters.RenderedVoxelGrid.mu
    gt_vox.set_origin(gt_vox.origin)

    frame_data = paths.RenderedData.load_scene_data(
        sequence['scene'], sequence['frames'][0])
    im = images.RGBDImage.load_from_dict(
        paths.RenderedData.scene_dir(sequence['scene']), frame_data)
    norm_engine = features.Normals()
    im.normals = norm_engine.compute_normals(im)

    logging.debug("Sampling from image")
    idxs = im.random_sample_from_mask(
        parameters.VoxletTraining.number_points_from_each_image)

    logging.debug("Voxel carving")
    video = images.RGBDVideo.init_from_images([im])
    carver = carving.Fusion()
    carver.set_video(video)
    carver.set_voxel_grid(gt_vox)
    im_tsdf, visible = carver.fuse(mu=parameters.RenderedVoxelGrid.mu)
    im_tsdf.V[np.isnan(im_tsdf.V)] = -parameters.RenderedVoxelGrid.mu

    logging.debug("Extracting shoeboxes and features...")
    t1 = time()
    gt_shoeboxes = [voxlets.extract_single_voxlet(
        idx, im=im, vgrid=gt_vox, post_transform=pca_flatten) for idx in idxs]
    view_shoeboxes = [voxlets.extract_single_voxlet(
        idx, im=im, vgrid=im_tsdf, post_transform=feature_transform) for idx in idxs]
    logging.debug("\n-> ...Took %f s" % (time() - t1))

    np_sboxes = np.vstack(gt_shoeboxes)
    np_features = np.vstack(view_shoeboxes)

    logging.debug("...Shoeboxes are shape " + str(np_sboxes.shape))
    logging.debug("...Features are shape " + str(np_features.shape))

    savepath = paths.RenderedData.voxlets_data_path + \
        sequence['name'] + '.mat'
    logging.debug("Saving to " + savepath)
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
