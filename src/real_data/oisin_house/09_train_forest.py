'''
train the model dammit
'''
import numpy as np
import cPickle as pickle
import sys
import os
import system_setup
import time
import yaml
import gc

parameters_path = './training_params_nyu.yaml'
parameters = yaml.load(open(parameters_path))

if parameters['training_data'] == 'oisin_house':
    import real_data_paths as paths
elif parameters['training_data'] == 'synthetic':
    import synthetic_paths as paths
elif parameters['training_data'] == 'nyu_cad':
    import nyu_cad_paths as paths
else:
    raise Exception('Unknown training data')

sys.path.append(os.path.expanduser('~/projects/shape_sharing/src/'))
from common import voxlets

if system_setup.small_sample:
    print "WARNING: Just computing on a small sample"


def load_training_data(voxlet_name, feature_name, num_scenes=None):
    '''
    Loading in all the data...
    '''
    features = []
    pca_representation = []
    masks = []
    scene_ids = []

    if num_scenes is not None:
        scenes_to_use = paths.all_train_data[:num_scenes]
    else:
        scenes_to_use = paths.all_train_data

    for count, sequence in enumerate(scenes_to_use):

        loadfolder = paths.voxlets_data_path % voxlet_name
        loadpath = loadfolder + sequence['name'] + '.pkl'
        if not os.path.exists(loadpath):
            print "Cannot find ", sequence['name']
            print "SKIPPING"
            continue

        D = pickle.load(open(loadpath, 'r'))

        features.append(D[feature_name])
        pca_representation.append(D['shoeboxes'])
        masks.append(D['masks'])
        scene_ids.append(np.ones(D['cobweb'].shape[0]) * count)

    np_voxlets = np.vstack(pca_representation)
    np_masks = np.vstack(masks)
    np_features = np.concatenate(features, axis=0)
    np_scene_ids = np.concatenate(scene_ids, axis=0).astype(int)

    print "\tVoxlets is\t", np_voxlets.shape
    print "\tMasks is\t", np_masks.shape
    print "\tFeature type is\t", feature_name
    print "\tFeatures is\t", np_features.shape
    print "\tThere are %d nan features" % np.isnan(np_features).sum()
    print "\tScene ids is\t", np_scene_ids.shape

    np_features[np.isnan(np_features)] = \
        float(parameters[feature_name + '_out_of_range_feature'])

    return np_features, np_voxlets, np_masks, np_scene_ids


def train_model(model_params, all_params):

    print "-> Ensuring output folder exists"
    savepath = paths.voxlet_model_path % (model_params['name'])

    modelfolder = os.path.dirname(savepath)
    if not os.path.exists(modelfolder):
        os.makedirs(modelfolder)

    print "-> Loading training data"
    if 'num_scenes' in model_params:
        print ">>> Subsampling to %d scenes!" % model_params['num_scenes']
        np_features, np_voxlets, np_masks, np_scene_ids = \
            load_training_data(model_params['voxlet_type'],
                model_params['feature'], model_params['num_scenes'])
    else:
        np_features, np_voxlets, np_masks, np_scene_ids = \
            load_training_data(model_params['voxlet_type'],
                model_params['feature'])

    print "-> Training forest"
    voxlet_params = parameters['voxlet_sizes'][model_params['voxlet_type']]

    model = voxlets.VoxletPredictor()
    model.set_voxlet_params(voxlet_params)
    model.train(
        np_features,
        np_voxlets,
        ml_type=parameters['ml_type'],
        forest_params=parameters['forest'],
        subsample_length=parameters['forest']['subsample_length'],
        masks=np_masks,
        scene_ids=np_scene_ids)
    model.feature = model_params['feature']
    print model.feature

    print "-> Adding PCA models"
    pca_savefolder = paths.voxlets_dictionary_path % voxlet_params['name']
    pca = pickle.load(open(pca_savefolder + 'voxlets_pca.pkl'))
    mask_pca = pickle.load(open(pca_savefolder + 'masks_pca.pkl'))

    model.set_pca(pca)
    model.set_masks_pca(mask_pca)

    # saving all the parameters for things like feature parameters etc
    model.all_params = all_params

    print "-> Saving to ", savepath
    model.save(savepath.replace('.pkl', '_full.pkl'))
    if hasattr(model, 'forest'):
        model.forest.make_lightweight()
    model.save(savepath)

    gc.collect()


if __name__ == '__main__':

    # Repeat for each type of voxlet in the parameters
    for model_params in parameters['models_to_train']:
        train_model(model_params, parameters)
        gc.collect()
            # del model, pca, mask_pca, np_features, np_voxlets, np_masks
