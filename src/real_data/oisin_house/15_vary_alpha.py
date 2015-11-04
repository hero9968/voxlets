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
import scipy.io
from copy import deepcopy

from common import voxlets, scene
import system_setup

if len(sys.argv) > 1:
    parameters_path = sys.argv[1]
else:
    parameters_path = './testing_params_nyu.yaml'
parameters = yaml.load(open(parameters_path))

if parameters['testing_data'] == 'oisin_house':
    import real_data_paths as paths
elif parameters['testing_data'] == 'synthetic':
    import synthetic_paths as paths
elif parameters['testing_data'] == 'nyu_cad':
    import nyu_cad_paths as paths
else:
    raise Exception('Unknown training data')

render_top_view = True
#
# print "WAITING"
# from time import sleep
# sleep(1000*3600)

if __name__ == '__main__':

    # loop over each test time in the testing parameters:
    params = [xx for xx in parameters['tests']
        if xx['name'] == 'short_samples_no_segment'][0]

    # print "--> DOING TEST: ", params['name']

    print "--> Loading models..."
    # print "WARNING"
    # vox_model_path = '/media/ssd/data/rendered_arrangements/models/%s/model.pkl'
    vox_model_path = paths.voxlet_model_path
    print [vox_model_path % name for name in params['models_to_use']]
    models = [pickle.load(open(vox_model_path % name))
              for name in params['models_to_use']]

    for alpha in [10000]:#0.0, 5, 10, 50, 100, 500, 1000]:
    # for weight_empty_lower in np.range(0, 1, 0.1):

        print "-->> Alpha is ", alpha
        # print
        params['reconstruction_params']['weight_parameter'] = alpha

        def process_sequence(sequence):

            print "-> Loading ", sequence['name']
            sc = scene.Scene(params['mu'], [])
            sc.load_sequence(
                sequence, frame_nos=0, segment_with_gt=False, segment=False)
            sc.sample_points(params['number_samples'],
                nyu='nyu_cad' in parameters['testing_data'])
            sc.im._clear_cache()

            print "-> Creating folder"
            fpath = paths.prediction_folderpath % \
                (parameters['batch_name'], sequence['name'])
            if not os.path.exists(fpath):
                os.makedirs(fpath)

            print "-> Setting up the reconstruction object"
            rec = voxlets.Reconstructer()
            rec.set_scene(sc)
            rec.initialise_output_grid(gt_grid=sc.gt_tsdf,
                keep_explicit_count=params['reconstruction_params']['weight_predictions'])
            rec.set_model_probabilities(params['model_probabilities'])
            rec.set_model(models)
            rec.mu = params['mu']

            for model in rec.model:
                model.reset_voxlet_counts()
                model.set_max_depth(params['max_depth'])
            print "-> Doing prediction, type ", params['name']
            # parameters from the yaml file are passed as separate arguments to voxlets
            pred_voxlets = rec.fill_in_output_grid(**params['reconstruction_params'])

            prediction_savepath = fpath + params['name'] + ('_%f_alpha.pkl' % alpha)
            print "-> Saving the prediction to ", prediction_savepath

            with open(prediction_savepath, 'w') as f:
                pickle.dump(pred_voxlets, f, protocol=pickle.HIGHEST_PROTOCOL)

        tic = time()

        if False:#system_setup.multicore:
            # need to import this *after* the pool helper has been defined
            import multiprocessing
            pool = multiprocessing.Pool(system_setup.testing_cores)
            pool.map_async(process_sequence, paths.test_data).get(9999999)
            pool.close()
            pool.join()
        else:
            map(process_sequence, paths.test_data)

        print "This test took %f s" % (time() - tic)
