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


def flatten_sbox(sbox):
    return sbox.V.flatten()

with open(paths.train_location, 'r') as f:
    train_objects = [l.strip() for l in f]

with open(paths.poses_to_use, 'r') as f:
    poses = [l.strip() for l in f]

from copy import deepcopy


# now creating some sequences...
train_data = []
for train_object in train_objects:

    sequence= {}
    sequence['scene'] = train_object
    sequence['frames'] = 'ss'
    sequence['folder'] = '/media/ssd/data/bigbird_cropped/'
    fp = '/media/ssd/data/bigbird_meshes/%s/meshes/voxelised.vox' % sequence['scene']

    grid = voxel_data.WorldVoxels()
    grid.populate_from_vox_file(fp)

    for pose in poses[::3]:

        sequence['name'] = train_object + '_' + pose
        sequence['pose_id'] = pose
        print "Processing " + sequence['scene'] + pose

        sc = scene.Scene(parameters.mu, voxlets.voxlet_class_to_dict(parameters.Voxlet))
        sc.load_bb_sequence(sequence)
        sc.gt_tsdf = deepcopy(grid)
        sc.gt_tsdf_separate = grid.V


        # just using the reconstructor for its point sampling routine!
        rec = voxlets.Reconstructer(
            reconstruction_type='kmeans_on_pca',combine_type='modal_vote')
        rec.set_scene(sc)
        rec.sample_points(100,
                          parameters.VoxletPrediction.sampling_grid_size,
                          additional_mask=sc.gt_im_label != 0)
        idxs = rec.sampled_idxs

        "Now try to make this nice and like parrallel or something...?"
        print "Extracting voxlets"
        t1 = time()
        try:
            temp_shoeboxes = [sc.extract_single_voxlet(
                idx, extract_from='actual_tsdf', post_transform=None) for idx in idxs]
        except:
            print "FAILED on " + sequence['scene']
            continue

        gt_shoeboxes = []
        for temp_shoebox in temp_shoeboxes:
            temp_shoebox.convert_to_tsdf(0.03)
            gt_shoeboxes.append(temp_shoebox.V.flatten())

        np_gt_sboxes = pca.transform(gt_shoeboxes)

        cobwebengine.set_image(sc.im)
        np_cobweb = np.array(cobwebengine.extract_patches(idxs))


        print "Took %f s" % (time() - t1)

        print "Shoeboxes are shape " + str(np_gt_sboxes.shape)

        savepath = paths.voxlets_data_path + \
            sequence['name'] + '.pkl'
        logging.debug("Saving to " + savepath)
        D = dict(shoeboxes=np_gt_sboxes, cobweb=np_cobweb)
        with open(savepath, 'w') as f:
            pickle.dump(D, f, protocol=pickle.HIGHEST_PROTOCOL)


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
