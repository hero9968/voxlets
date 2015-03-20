'''
a script to set up the folders for the training data, also to do things like
the partial kinfu perhaps...
and I think now this should actually do the full thing to extract and save
training data from each sequence
'''

import sys
import os
import scipy.io
from time import time

sys.path.append(os.path.expanduser('~/projects/shape_sharing/src/'))
sys.path.append(os.path.expanduser('~/projects/shape_sharing/src/intrinsic/'))

from common import paths
from common import parameters
from common import scene
from features import line_casting


def process_sequence(sequence):

    seq_foldername = paths.RenderedData.implicit_training_dir % sequence['scene']

    if os.path.exists(seq_foldername):
        print "Saving to %s" % seq_foldername
    else:
        print "Creating %s" % seq_foldername
        os.makedirs(seq_foldername)

    print "Processing " + sequence['name']
    sc = scene.Scene()
    sc.load_sequence(sequence, frame_nos=0, segment_with_gt=False, segment=False, save_grids=False)

    # getting the known full and empty voxels based on the depth image
    known_full_voxels = sc.im_visible
    known_empty_voxels = sc.im_tsdf.blank_copy()
    known_empty_voxels.V = sc.im_tsdf.V > 0

    X, Y = line_casting.feature_pairs_3d(
        known_empty_voxels, known_full_voxels, sc.gt_tsdf.V,
        samples=10000, base_height=0, autorotate=False)

    training_pairs = dict(X=X, Y=Y)
    scipy.io.savemat(seq_foldername + 'training_pairs.mat', training_pairs)



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
