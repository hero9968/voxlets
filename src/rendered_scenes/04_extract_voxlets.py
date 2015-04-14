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

import paths
import parameters

sys.path.append(os.path.expanduser('~/projects/shape_sharing/src/'))
from common import voxel_data
from common import images
from common import features
from common import carving
from common import voxlets
from common import scene

if not os.path.exists(paths.RenderedData.voxlets_dict_data_path):
    os.makedirs(paths.RenderedData.voxlets_dict_data_path)


def flatten_sbox(sbox):
    return sbox.V.flatten()


def process_sequence(sequence):

    print "Processing " + sequence['name']
    sc = scene.Scene(parameters.RenderedVoxelGrid.mu, parameters.Voxlet)
    sc.load_sequence(sequence, frame_nos=0, segment_with_gt=True, save_grids=False)
    # sc.santity_render(save_folder='/tmp/')

    # just using the reconstructor for its point sampling routine!
    rec = voxlets.Reconstructer(
        reconstruction_type='kmeans_on_pca', combine_type='modal_vote')
    rec.set_scene(sc)
    rec.sample_points(parameters.VoxletTraining.pca_number_points_from_each_image,
                      parameters.VoxletPrediction.sampling_grid_size,
                      additional_mask=sc.gt_im_label != 0)
    idxs = rec.sampled_idxs

    "Now try to make this nice and like parrallel or something...?"
    t1 = time()
    gt_shoeboxes = [sc.extract_single_voxlet(
        idx, extract_from='gt_tsdf', post_transform=flatten_sbox) for idx in idxs]
    view_shoeboxes = [sc.extract_single_voxlet(
        idx, extract_from='im_tsdf', post_transform=flatten_sbox) for idx in idxs]

    np_gt_sboxes = np.array(gt_shoeboxes)
    np_features = np.array(view_shoeboxes)
    print "View sboxes are shape", np_features.shape

    print "Took %f s" % (time() - t1)

    print "Shoeboxes are shape " + str(np_gt_sboxes.shape)
    print "Features are shape " + str(np_features.shape)

    D = dict(shoeboxes=np_gt_sboxes, features=np_features)
    savepath = paths.RenderedData.voxlets_dict_data_path + \
        sequence['name'] + '.mat'
    print savepath
    scipy.io.savemat(savepath, D, do_compression=True)


# need to import these *after* the pool helper has been defined
if parameters.multicore:
    import multiprocessing
    import functools
    pool = multiprocessing.Pool(parameters.cores)
    mapper = pool.map
else:
    mapper = map


if __name__ == '__main__':

    tic = time()
    mapper(process_sequence, paths.RenderedData.train_sequence())
    print "In total took %f s" % (time() - tic)
