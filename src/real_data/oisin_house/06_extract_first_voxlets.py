'''
Loads in the combined sboxes and clusters them to form a dictionary
smaller
'''
import numpy as np
import cPickle as pickle
import sys
import os
from time import time
import scipy.io

sys.path.append(os.path.expanduser('~/projects/shape_sharing/src/'))
from common import voxlets, scene

import real_data_paths as paths

if not os.path.exists(paths.voxlets_dict_data_path):
    os.makedirs(paths.voxlets_dict_data_path)

parameters_path = './training_params.yaml'
parameters = yaml.load(open(parameters_path))


def flatten_sbox(sbox):
    return sbox.V.flatten()


def process_sequence(sequence):

    if not os.path.exists(sequence['folder'] + sequence['scene'] + '/ground_truth_tsdf.pkl'):
        print "Failed"
        return
    # try:
    print "Processing " + sequence['scene']
    sc = scene.Scene(parameters['mu'], voxlets.voxlet_class_to_dict(parameters.Voxlet))
    sc.load_sequence(sequence, frame_nos=0, segment_with_gt=True,
        save_grids=False, voxel_normals=True)

    # just using the reconstructor for its point sampling routine!
    rec = voxlets.Reconstructer(
        reconstruction_type='kmeans_on_pca', combine_type='modal_vote')
    rec.set_scene(sc)
    rec.sample_points(parameters['pca_number_points_from_each_image'],
                      parameters['sampling_grid_size'],
                      additional_mask=sc.gt_im_label != 0)
    idxs = rec.sampled_idxs

    print "-> Extracting voxlets"
    t1 = time()
    gt_shoeboxes = [sc.extract_single_voxlet(
        idx, extract_from='gt_tsdf', post_transform=flatten_sbox) for idx in idxs]
    print "-> Took %f s" % (time() - t1)

    return np.array(gt_shoeboxes)


# need to import these *after* the pool helper has been defined
if  parameters.multicore:
    import multiprocessing
    import functools
    pool = multiprocessing.Pool(parameters.cores)
    mapper = pool.map
else:
    mapper = map


def extract_all_voxlets(voxlet_params):

    voxlets = mapper(process_sequence, paths.all_train_data[:3])
    print "length is ", len(voxlets)
    np_voxlets = np.vstack(voxlets)
    print "-> Shoeboxes are shape " + str(np_voxlets.shape)
    return np_voxlets



if __name__ == '__main__':


    for voxlet_params in parameters['voxlets']:
        tic = time()

        print "-> Extracting the voxlets, type %s" % voxlet_params['name']
        voxlets = extract_all_voxlets(voxlet_params)

        print "-> Doing the PCA"

        print "-> Saving the PCA"

        print "In total took %f s" % (time() - tic)
