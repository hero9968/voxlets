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
        print sc.im.rgb.shape

        # # just using the reconstructor for its point sampling routine!
        # rec = voxlets.Reconstructer(
        #     reconstruction_type='kmeans_on_pca',combine_type='modal_vote')
        # rec.set_scene(sc)
        # rec.sample_points(50,
        #                   parameters.VoxletPrediction.sampling_grid_size,
        #                   additional_mask=sc.gt_im_label != 0)
        # idxs = rec.sampled_idxs

        # "Now try to make this nice and like parrallel or something...?"
        # print "Extracting voxlets"
        # t1 = time()
        # try:
        #     temp_shoeboxes = [sc.extract_single_voxlet(
        #         idx, extract_from='actual_tsdf', post_transform=None) for idx in idxs]
        # except:
        #     print "FAILED on " + sequence['scene']
        #     continue

        # gt_shoeboxes = []
        # for temp_shoebox in temp_shoeboxes:
        #     temp_shoebox.convert_to_tsdf(0.03)
        #     gt_shoeboxes.append(temp_shoebox.V.flatten())

        # np_gt_sboxes = np.vstack(gt_shoeboxes)
        # print np.isnan(np_gt_sboxes.flatten()).sum() / float(np_gt_sboxes.size)
        # if np.all(np.isnan(np_gt_sboxes.flatten())):
        #     print "ALL nan"
        #     dsdf


        # print "Took %f s" % (time() - t1)

        # print "Shoeboxes are shape " + str(np_gt_sboxes.shape)

        # D = dict(shoeboxes=np_gt_sboxes)
        # savepath = paths.voxlets_dict_data_path + \
        #     sequence['name'] + '.pkl'
        # with open(savepath, 'w') as f:
        #     pickle.dump(D, f, protocol=pickle.HIGHEST_PROTOCOL)
        # print savepath


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
