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
from common import scene, voxlets

import real_data_paths as paths
import real_params as parameters

# features_iso_savepath = paths.RenderedData.voxlets_dictionary_path + 'features_iso.pkl'
# with open(features_iso_savepath, 'rb') as f:
#     features_iso = pickle.load(f)

pca_savepath = paths.voxlets_dictionary_path + 'shoeboxes_pca.pkl'
with open(pca_savepath, 'rb') as f:
    pca = pickle.load(f)

mask_pca_savepath = paths.voxlets_dictionary_path + 'masks_pca.pkl'
with open(mask_pca_savepath, 'rb') as f:
    mask_pca = pickle.load(f)

features_pca_savepath = paths.voxlets_dictionary_path + 'features_pca.pkl'
with open(features_pca_savepath, 'rb') as f:
    features_pca = pickle.load(f)


print "PCA components is shape ", pca.components_.shape
print "Features PCA components is shape ", features_pca.components_.shape

if not os.path.exists(paths.voxlets_data_path):
    os.makedirs(paths.voxlets_data_path)


def decimate_flatten(sbox):
    return sbox.V[::2, ::2, ::2].flatten()


def pca_flatten(sbox):
    """Applied to the GT shoeboxes after extraction"""
    return pca.transform(sbox.V.flatten())


def sbox_flatten(sbox):
    """Applied to the GT shoeboxes after extraction"""
    return sbox.V.flatten()

logf = open('/home/michael/Desktop/log.txt', 'w')

def process_sequence(sequence):

    logging.info("Processing " + sequence['name'])

    try:
        sc = scene.Scene(parameters.mu, voxlets.voxlet_class_to_dict(parameters.Voxlet))
        sc.load_sequence(sequence, frame_nos=0, segment_with_gt=True,
            save_grids=False, load_implicit=False, voxel_normals='gt_tsdf')
        # sc.santity_render(save_folder='/tmp/')

        # just using reconstructor for sampling the points...
        rec = voxlets.Reconstructer(
            reconstruction_type='kmeans_on_pca', combine_type='modal_vote')
        rec.set_scene(sc)
        rec.sample_points(parameters.VoxletTraining.number_points_from_each_image,
                          parameters.VoxletPrediction.sampling_grid_size,
                          additional_mask=sc.gt_im_label != 0)
        idxs = rec.sampled_idxs

        logging.debug("Extracting shoeboxes and features...")
        t1 = time()
        gt_shoeboxes = [sc.extract_single_voxlet(
            idx, extract_from='gt_tsdf', post_transform=sbox_flatten) for idx in idxs]

        if parameters.VoxletTraining.feature_transform == 'pca':

            view_shoeboxes = [sc.extract_single_voxlet(
                idx, extract_from='im_tsdf', post_transform=sbox_flatten) for idx in idxs]
            all_features = np.vstack(view_shoeboxes)
            all_features[np.isnan(all_features)] = -parameters.mu
            np_features = features_pca.transform(all_features)

        elif parameters.VoxletTraining.feature_transform == 'decimate':

            view_shoeboxes = [sc.extract_single_voxlet(
                idx, extract_from='im_tsdf', post_transform=decimate_flatten) for idx in idxs]
            np_features = np.vstack(view_shoeboxes)
            np_features[np.isnan(np_features)] = -parameters.mu

        np_sboxes = np.vstack(gt_shoeboxes)

        # Doing the mask trick...
        np_masks = np.isnan(np_sboxes).astype(np.float16)
        np_sboxes[np_masks == 1] = np.nanmax(np_sboxes)

        # must do the pca now after doing the mask trick
        np_sboxes = pca.transform(np_sboxes)
        np_masks = mask_pca.transform(np_masks)

        '''replace all the nans in the shoeboxes from the image view'''


        logging.debug("...Shoeboxes are shape " + str(np_sboxes.shape))
        logging.debug("...Features are shape " + str(np_features.shape))

        print "Took %f s" % (time() - t1)
        t1 = time()

        savepath = paths.voxlets_data_path + \
            sequence['name'] + '.pkl'
        logging.debug("Saving to " + savepath)
        D = dict(shoeboxes=np_sboxes, features=np_features, masks=np_masks)
        with open(savepath, 'w') as f:
            pickle.dump(D, f, protocol=pickle.HIGHEST_PROTOCOL)
    except:
        print "FAILED"
        logf.write(sequence['name'])
        logf.write('\n')


if parameters.multicore:
    # need to import these *after* pool_helper has been defined
    import multiprocessing
    pool = multiprocessing.Pool(parameters.cores)
    mapper = pool.map
else:
    mapper = map


if __name__ == "__main__":

    tic = time()
    mapper(process_sequence, paths.train_data)
    print "In total took %f s" % (time() - tic)

logf.close()