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
from common import parameters
from common import scene

# features_iso_savepath = paths.RenderedData.voxlets_dictionary_path + 'features_iso.pkl'
# with open(features_iso_savepath, 'rb') as f:
#     features_iso = pickle.load(f)

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


def sbox_flatten(sbox):
    """Applied to the GT shoeboxes after extraction"""
    return sbox.V.flatten()



def process_sequence(sequence):

    logging.info("Processing " + sequence['name'])

    sc = scene.Scene()
    sc.load_sequence(sequence, frame_nos=0, segment_with_gt=True,
        save_grids=False, load_implicit=parameters.VoxletTraining.use_implicit)
    sc.santity_render(save_folder='/tmp/')

    idxs = sc.im.random_sample_from_mask(
        parameters.VoxletTraining.number_points_from_each_image)

    logging.debug("Extracting shoeboxes and features...")
    t1 = time()
    gt_shoeboxes = [sc.extract_single_voxlet(
        idx, extract_from='gt_tsdf', post_transform=pca_flatten) for idx in idxs]
    view_shoeboxes = [sc.extract_single_voxlet(
        idx, extract_from='visible_tsdf', post_transform=sbox_flatten) for idx in idxs]

    np_sboxes = np.vstack(gt_shoeboxes)
    np_view = np.vstack(view_shoeboxes)

    '''replace all the nans in the shoeboxes from the image view'''
    np_view[np.isnan(np_view)] = -parameters.RenderedVoxelGrid.mu

    logging.debug("...Shoeboxes are shape " + str(np_sboxes.shape))
    logging.debug("...Features are shape " + str(np_view.shape))

    if parameters.VoxletTraining.use_implicit:
        implicit_sboxes = [sc.extract_single_voxlet(
            idx, extract_from='implicit_tsdf', post_transform=sbox_flatten) for idx in idxs]
        np_implicit = np.vstack(implicit_sboxes)
        logging.debug("...Implicit is shape " + str(np_implicit.shape))
        all_features = np.concatenate((np_view, np_implicit), axis=1)
        print np.isnan(np_view).sum(), np.isnan(np_implicit).sum()

    else:
        all_features = np_view

    print "Took %f s" % (time() - t1)
    t1 = time()

    np_features = features_pca.transform(all_features)

    # iso_features = features_iso.transform(all_features)
    # print "iso features is shape ", iso_features.shape
    # print "Transform took %f s" % (time() - t1)

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
