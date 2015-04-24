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
from common import scene, voxlets, features

import real_data_paths as paths
import real_params as parameters

# features_iso_savepath = paths.RenderedData.voxlets_dictionary_path + 'features_iso.pkl'
# with open(features_iso_savepath, 'rb') as f:
#     features_iso = pickle.load(f)

pca_savepath = paths.voxlets_dictionary_path + 'shoeboxes_pca.pkl'
with open(pca_savepath, 'rb') as f:
    pca = pickle.load(f)

print "PCA components is shape ", pca.components_.shape

if not os.path.exists(paths.voxlets_data_path):
    os.makedirs(paths.voxlets_data_path)

cobwebengine = features.CobwebEngine(0.075, mask=True)

def decimate_flatten(sbox):
    return sbox.V[::2, ::2, ::2].flatten()


def pca_flatten(sbox):
    """Applied to the GT shoeboxes after extraction"""
    return pca.transform(sbox.V.flatten())


def sbox_flatten(sbox):
    """Applied to the GT shoeboxes after extraction"""
    return sbox.V.flatten()

logf = open('/home/michael/Desktop/log.txt', 'w')

cobwebengine = features.CobwebEngine(0.01, mask=True)
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
from common import voxlets
from common import scene
from common import voxel_data

import real_data_paths as paths
import real_params as parameters

if not os.path.exists(paths.voxlets_dict_data_path):
    os.makedirs(paths.voxlets_dict_data_path)

prediction_type = 'first_round'


def flatten_sbox(sbox):
    return sbox.V.flatten()

with open(paths.train_location, 'r') as f:
    train_objects = [l.strip() for l in f]

with open(paths.poses_to_use, 'r') as f:
    poses = [l.strip() for l in f]

from copy import deepcopy

with open(paths.voxlet_model_oma_path.replace('.pkl', '_cobweb.pkl'), 'rb') as f:
    model = pickle.load(f)

rec = voxlets.Reconstructer(
    reconstruction_type='kmeans_on_pca',combine_type='modal_vote')
rec.set_model(model)

fpath = paths.voxlet_prediction_folderpath % prediction_type
if not os.path.exists(fpath):
    os.makedirs(fpath)

# now creating some sequences...
train_data = []
for train_object in train_objects:

    sequence= {}
    sequence['scene'] = train_object
    sequence['frames'] = 'ss'
    sequence['folder'] = '/media/ssd/data/bigbird_cropped/'
    fp = '/media/ssd/data/bigbird_meshes/%s/meshes/voxelised.vox' % sequence['scene']

    gt_grid = voxel_data.WorldVoxels()
    gt_grid.populate_from_vox_file(fp)
    gt_grid.convert_to_tsdf(0.03)

    for pose in poses[::3]:

        sequence['name'] = train_object + '_' + pose
        sequence['pose_id'] = pose
        print "Processing " + sequence['scene'] + pose

        sc = scene.Scene(parameters.mu, voxlets.voxlet_class_to_dict(parameters.Voxlet))
        sc.load_bb_sequence(sequence)
        sc.gt_tsdf = gt_grid

        # just using the reconstructor for its point sampling routine!
        rec.set_scene(sc)
        rec.initialise_output_grid(gt_grid=gt_grid)
        sc.norrms()

        found_it = False
        for i in range(100):
            rec.sample_points(250,
                              parameters.VoxletPrediction.sampling_grid_size,
                              additional_mask=sc.gt_im_label != 0)

            norms = sc.im.get_world_normals()

            found_it = True

            for index in rec.sampled_idxs:
                point_idx = index[0] * sc.im.mask.shape[1] + index[1]
                these_norms = norms[point_idx, :]
                if np.any(np.abs(these_norms)) == 1 or np.any(np.isnan(these_norms)):
                    found_it = False

            if found_it:
                break





        if found_it:
            pred_voxlets = rec.fill_in_output_grid_oma(
                add_ground_plane=False, render_type=[],
                weight_empty_lower=None, cobweb=True)

            savepath = paths.voxlet_prediction_savepath % \
                (prediction_type, sequence['name'] + '_' + sequence['pose_id'] + '.mat')
            D = dict(prediction = pred_voxlets.V, gt = gt_grid.V)
            scipy.io.savemat(savepath, D)
        else:
            print "Count not do it"

# # need to import these *after* the pool helper has been defined
# if False:
#  # parameters.multicore:
#     import multiprocessing
#     import functools
#     pool = multiprocessing.Pool(parameters.cores)
#     mapper = pool.map
# else:
#     mapper = map


# if __name__ == '__main__':

#     tic = time()
#     mapper(process_sequence, paths.train_data)
#     print "In total took %f s" % (time() - tic)
