
import numpy as np
import cPickle as pickle
import sys
import os
import yaml
import system_setup
import scipy.misc
import matplotlib.pyplot as plt
sys.path.append(os.path.expanduser("~/projects/shape_sharing/src/"))
from common import voxlets, scene

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
elif parameters['testing_data'] == 'nyu_cad_silberman':
    import nyu_cad_paths_silberman as paths
else:
    raise Exception('Unknown training data')


def process_sequence(sequence):

    print "-> Loading ground truth", sequence['name']
    fpath = paths.prediction_folderpath % (parameters['batch_name'], sequence['name'])
    gt_scene = pickle.load(open(fpath + 'ground_truth.pkl'))

    # Path where any renders will be saved to
    gen_renderpath = paths.voxlet_prediction_img_path % \
        (parameters['batch_name'], sequence['name'], '%s')

    print "-> Main renders"
    for test_params in parameters['tests']:

            if test_params['name'] == 'ground_truth' or test_params['name'] == 'visible':
                continue

            print "Loading ", test_params['name']

            prediction_savepath = fpath + test_params['name'] + '.pkl'

            prediction = pickle.load(open(prediction_savepath))

            evaluation_region_loadpath = paths.evaluation_region_path % (
                parameters['batch_name'], sequence['name'])
                # pickle.dump(evaluation_region, open(savepath, 'w'), -1)
            evaluation_region = scipy.io.loadmat(
                evaluation_region_loadpath)['evaluation_region'] > 0

            result = gt_scene.evaluate_prediction(prediction.V,
                voxels_to_evaluate=evaluation_region)

            results_savepath = fpath + "../eval_%s.yaml" % test_params['name']
            yaml.dump(result, open(results_savepath, 'w'))


# need to import these *after* the pool helper has been defined
if system_setup.multicore:
    import multiprocessing
    mapper = multiprocessing.Pool(system_setup.testing_cores).map
else:
    mapper = map


if __name__ == '__main__':

    # print "WARNING - SMALL TEST DATA"
    # test_data = yaml.load(open('/media/ssd/data/oisin_house/train_test/test.yaml'))
    test_data = paths.test_data
    results = mapper(process_sequence, test_data)
