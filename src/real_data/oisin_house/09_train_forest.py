'''
train the model dammit
'''
import numpy as np
import cPickle as pickle
import sys
import os
import system_setup
import real_data_paths as paths
import time
import yaml

sys.path.append(os.path.expanduser('~/projects/shape_sharing/src/'))
from common import voxlets

if system_setup.small_sample:
    print "WARNING: Just computing on a small sample"


parameters_path = './training_params.yaml'
parameters = yaml.load(open(parameters_path))


def load_training_data(voxlet_params):
    '''
    Loading in all the data...
    '''
    features = []
    pca_representation = []
    masks = []
    scene_ids = []

    for count, sequence in enumerate(paths.all_train_data):

        loadfolder = paths.voxlets_data_path % voxlet_params['name']
        loadpath = loadfolder + sequence['name'] + '.pkl'
        D = pickle.load(open(loadpath, 'r'))

        features.append(D['cobweb'])
        pca_representation.append(D['shoeboxes'])
        masks.append(D['masks'])
        scene_ids.append(np.ones(D['cobweb'].shape[0]) * count)

    np_voxlets = np.vstack(pca_representation)
    np_masks = np.vstack(masks)
    np_features = np.concatenate(features, axis=0)
    np_scene_ids = np.concatenate(scene_ids, axis=0).astype(int)

    print "\tVoxlets is\t", np_voxlets.shape
    print "\tMasks is\t", np_masks.shape
    print "\tFeatures is\t", np_features.shape
    print "\tScene ids is\t", np_scene_ids.shape

    np_features[np.isnan(np_features)] = float(parameters['out_of_range_feature'])

    return np_features, np_voxlets, np_masks, np_scene_ids


if __name__ == '__main__':

    # Repeat for each type of voxlet in the parameters
    for voxlet_params in parameters['voxlets']:

        print "-> Ensuring output folder exists"
        savepath = paths.voxlet_model_path % voxlet_params['name']

        modelfolder = os.path.dirname(savepath)
        if not os.path.exists(modelfolder):
            os.makedirs(modelfolder)

        print "-> Loading training data"
        np_features, np_voxlets, np_masks, np_scene_ids = \
            load_training_data(voxlet_params)

        print "-> Training forest"
        model = voxlets.VoxletPredictor()
        model.set_voxlet_params(voxlet_params)
        model.train(
            np_features,
            np_voxlets,
            forest_params=parameters['forest'],
            subsample_length=parameters['forest_subsample_length'],
            masks=np_masks,
            scene_ids=np_scene_ids)

        print "-> Adding PCA models"
        pca_savefolder = paths.voxlets_dictionary_path % voxlet_params['name']
        pca = pickle.load(open(pca_savefolder + 'voxlets_pca.pkl'))
        mask_pca = pickle.load(open(pca_savefolder + 'masks_pca.pkl'))

        model.set_pca(pca)
        model.set_masks_pca(mask_pca)

        print "-> Saving to ", savepath
        model.save(savepath)
