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
import yaml

sys.path.append(os.path.expanduser('~/projects/shape_sharing/src/'))

import system_setup
import real_data_paths as paths
from common import scene
from features import line_casting


parameters = yaml.load(open('./implicit_params.yaml'))
modelname = parameters['modelname']

print "Creating output folder"
savefolder = paths.implicit_training_dir % modelname
if not os.path.exists(savefolder):
    os.makedirs(savefolder)

def process_sequence(sequence):

    save_location = savefolder + sequence['name'] + '.mat'

    print "Processing " + sequence['name']
    sc = scene.Scene(parameters['mu'], None)
    sc.load_sequence(sequence, frame_nos=0, segment_with_gt=False,
        segment=False, save_grids=False, carve=True)

    # getting the known full and empty voxels based on the depth image
    known_full_voxels = sc.im_visible
    known_empty_voxels = sc.im_tsdf.blank_copy()
    known_empty_voxels.V = sc.im_tsdf.V > 0

    X, Y = line_casting.feature_pairs_3d(
        known_empty_voxels, known_full_voxels, sc.gt_tsdf.V,
        samples=parameters['training_samples_per_image'],
        base_height=0, postprocess=parameters['postprocess'])

    training_pairs = dict(X=X, Y=Y)
    scipy.io.savemat(save_location, training_pairs)


# need to import these *after* the pool helper has been defined
if system_setup.multicore:
    import multiprocessing
    mapper = multiprocessing.Pool(system_setup.cores).map
else:
    mapper = map


if __name__ == '__main__':

    tic = time()
    mapper(process_sequence, paths.all_train_data)
    print "In total took %f s" % (time() - tic)
