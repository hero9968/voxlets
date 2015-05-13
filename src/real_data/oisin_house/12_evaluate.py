
import numpy as np
import cPickle as pickle
import sys
import os
from time import time
import yaml
import real_data_paths as paths
import system_setup
sys.path.append(os.path.expanduser("~/projects/shape_sharing/src/"))
from common import voxlets, scene

parameters_path = './testing_params.yaml'
parameters = yaml.load(open(parameters_path))


def process_sequence(sequence):

    print "-> Loading ground truth", sequence['name']
    fpath = paths.prediction_folderpath % (parameters['batch_name'], sequence['name'])
    gt_scene = pickle.load(open(fpath + 'ground_truth.pkl'))

    results_dict = {}

    for test_params in parameters['tests']:

        prediction_savepath = fpath + test_params['name'] + '.pkl'
        if os.path.exists(prediction_savepath):

            print "loading ", prediction_savepath

            prediction = pickle.load(open(prediction_savepath))

            # sometimes multiple predictions are stored in predicton
            if hasattr(prediction, '__iter__'):
                for key, item in prediction.iteritems():
                    results_dict[test_params['name'] + str(key)] = \
                        gt_scene.evaluate_prediction(item.V)
            else:
                results_dict[test_params['name']] = \
                    gt_scene.evaluate_prediction(prediction.V)

        else:
            print "Could not load ", prediction_savepath

    fpath = paths.prediction_folderpath % \
        (parameters['batch_name'], sequence['name'])

    with open(fpath + 'scores.yaml', 'w') as f:
        f.write(yaml.dump(results_dict, default_flow_style=False))


# need to import these *after* the pool helper has been defined
if system_setup.multicore:
    import multiprocessing
    mapper = multiprocessing.Pool(system_setup.testing_cores).map
else:
    mapper = map


if __name__ == '__main__':

    mapper(process_sequence, paths.test_data)


            # results_dict[desc] = {
            #             'description': desc,
            #             'auc':         float(dic['auc']),
            #             'iou':         float(dic['iou']),
            #             'precision':   float(dic['precision']),
            #             'recall':      float(dic['recall'])}
