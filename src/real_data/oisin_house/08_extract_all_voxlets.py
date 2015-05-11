'''
Extracts all the shoeboxes from all the training images
'''
import numpy as np
import cPickle as pickle
import sys
import os
from time import time
import system_setup
import yaml

sys.path.append(os.path.expanduser('~/projects/shape_sharing/src/'))
from common import scene, voxlets, features

import real_data_paths as paths

parameters_path = './training_params.yaml'
parameters = yaml.load(open(parameters_path))


if not os.path.exists(paths.voxlets_data_path):
    os.makedirs(paths.voxlets_data_path)


def sbox_flatten(sbox):
    """Applied to the GT shoeboxes after extraction"""
    return sbox.V.flatten()


# where to log the failures
logf = open('/home/michael/Desktop/failure_log.txt', 'w')

cobwebengine = features.CobwebEngine(parameters['cobweb_offset'], mask=True)


def process_sequence(sequence, pca, mask_pca, voxlet_params):

    print "-> Processing " + sequence['name']

    try:
        sc = scene.Scene(parameters.mu, voxlet_params)
        sc.load_sequence(sequence, frame_nos=0, segment_with_gt=True, voxel_normals='gt_tsdf')
    except:
        print "FAILED"
        logf.write(sequence['name'] + '\n')
        return

    # just using reconstructor for sampling the points...
    rec = voxlets.Reconstructer()
    rec.set_scene(sc)
    rec.sample_points(parameters['number_points_from_each_image'],
                      parameters['sampling_grid_size'],
                      additional_mask=sc.gt_im_label != 0)
    idxs = rec.sampled_idxs

    print "-> Extracting shoeboxes and features..."
    t1 = time()
    gt_shoeboxes = [sc.extract_single_voxlet(
        idx, extract_from='gt_tsdf', post_transform=sbox_flatten) for idx in idxs]

    cobwebengine.set_image(sc.im)
    np_cobweb = np.array(cobwebengine.extract_patches(idxs))
    np_sboxes = np.vstack(gt_shoeboxes)

    # Doing the mask trick...
    np_masks = np.isnan(np_sboxes).astype(np.float16)
    np_sboxes[np_masks == 1] = np.nanmax(np_sboxes)

    # must do the pca now after doing the mask trick
    np_sboxes = pca.transform(np_sboxes)
    np_masks = mask_pca.transform(np_masks)

    print "...Shoeboxes are shape " + str(np_sboxes.shape))
    print "Took %f s" % (time() - t1)

    savepath = paths.voxlets_data_path + sequence['name'] + '.pkl'
    print "-> Saving to " + savepath

    D = dict(shoeboxes=np_sboxes, masks=np_masks, cobweb=np_cobweb)
    with open(savepath, 'w') as f:
        pickle.dump(D, f, protocol=pickle.HIGHEST_PROTOCOL)


if system_setup.multicore:
    # need to import these *after* pool_helper has been defined
    import multiprocessing
    mapper = multiprocessing.Pool(system_setup.cores).map
else:
    mapper = map


if __name__ == '__main__':

    # Repeat for each type of voxlet in the parameters
    for voxlet_params in parameters['voxlets']:

        print "Processing voxlet type :", voxlet_params['name']
        tic = time()

        print "-> Loading PCA models"
        pca_savepath = paths.voxlets_dictionary_path + \
            '_%s_voxlets_pca.pkl' % voxlet_params['name']
        pca = pickle.load(open(pca_savepath))

        savepath = paths.voxlets_dictionary_path + \
            '_%s_masks_pca.pkl' % voxlet_params['name']
        mask_pca = pickle.load(open(mask_pca_savepath))

        print "-> Extracting the voxlets, type %s" % voxlet_params['name']
        func = functools.partial(process_sequence,
            pca=pca, mask_pca=mask_pca, voxlet_params=voxlet_params)
        mapper(func, paths.all_train_data)

        print "In total took %f s" % (time() - tic)

logf.close()