'''
make predictions for all of bigbird dataset
using my algorhtm
saves each prediction to disk
'''
import numpy as np
import cPickle as pickle
import sys
import os
sys.path.append(os.path.expanduser("~/projects/shape_sharing/src/"))
from time import time
import yaml
import functools

import real_data_paths as paths
from common import voxlets, scene
import system_setup

parameters_path = './testing_params.yaml'
parameters = yaml.load(open(parameters_path))


def process_sequence(sequence, params, models):

    print "-> Loading ", sequence['name']
    sc = scene.Scene(params['mu'], [])
    sc.load_sequence(
        sequence, frame_nos=0, segment_with_gt=True, voxel_normals='gt_tsdf')
    sc.sample_points(params['number_samples'])

    print "-> Creating folder"
    fpath = paths.prediction_folderpath % \
        (parameters['batch_name'], sequence['name'])
    if not os.path.exists(fpath):
        os.makedirs(fpath)

    if 'ground_truth' in params and params['ground_truth']:
        pred_voxlets = sc.gt_tsdf

    elif 'visible' in params and params['visible']:
        pred_voxlets = sc.im_tsdf

    else:
        print "-> Setting up the reconstruction object"
        rec = voxlets.Reconstructer()
        rec.set_scene(sc)
        rec.initialise_output_grid(gt_grid=sc.gt_tsdf)
        rec.set_probability_model_one(0.5)
        rec.set_model(models)

        for model in rec.model:
            model.reset_voxlet_counts()
            model.set_max_depth(params['max_depth'])

        print "-> Doing prediction, type ", params['name']
        # parameters from the yaml file are passed as separate arguments to voxlets
        pred_voxlets = rec.fill_in_output_grid(**params['reconstruction_params'])

    prediction_savepath = fpath + params['name'] + '.pkl'
    print "-> Saving the prediction to ", prediction_savepath

    with open(prediction_savepath, 'w') as f:
        pickle.dump(pred_voxlets, f, protocol=pickle.HIGHEST_PROTOCOL)


# need to import these *after* the pool helper has been defined
if system_setup.multicore:
    import multiprocessing
    mapper = multiprocessing.Pool(system_setup.testing_cores).map
else:
    mapper = map


if __name__ == '__main__':

    # loop over each test time in the testing parameters:
    for test_params in parameters['tests']:

        print "--> DOING TEST: ", test_params['name']

        print "--> Loading models..."
        models = [pickle.load(open(paths.voxlet_model_path % name))
                  for name in test_params['models_to_use']]

        print "--> Doing test type ", test_params['name']
        tic = time()
        func = functools.partial(
            process_sequence, params=test_params, models=models)
        mapper(func, paths.test_data)
        print "This test took %f s" % (time() - tic)
