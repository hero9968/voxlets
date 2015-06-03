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
import numpy as np
sys.path.append(os.path.expanduser('~/projects/shape_sharing/src/'))
import cPickle as pickle
import system_setup
import paths
from common import scene, features

parameters = yaml.load(open('./params.yaml'))

print "Creating output folder"
savefolder = paths.features_dir
if not os.path.exists(savefolder):
    os.makedirs(savefolder)


fe = features.RegionFeatureEngine()
ne = features.Normals()

def process_sequence(sequence):

    save_location = savefolder + sequence['name'] + '.mat'

    print "Processing " + sequence['name']
    sc = scene.Scene(parameters['mu'], None)
    sc.load_sequence(sequence, frame_nos=0, segment_with_gt=True,
        segment=True, save_grids=False, carve=False)

    # computing normals
    # sc.im.normals = ne.compute_bilateral_normals(sc.im.depth)

    # we can get the labelling, which has been got from the ground truth image
    labelling = sc.gt_im_label
    labels = np.unique(labelling[~np.isnan(labelling)])
    region_features = {}

    for label in labels:
        # label 0 is probably the floor
        if label == 0:
            continue

        region_features[label] = \
            fe.compute_features(sc.im, labelling == label)

    # save the array of feature vector dicts to a file
    D = dict(features=region_features, labelling=labelling)
    # scipy.io.savemat(save_location, D)
    with open(save_location, 'w') as f:
        pickle.dump(D, f, protocol=pickle.HIGHEST_PROTOCOL)


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
