import matplotlib.pyplot as plt
import numpy as np
import cPickle as pickle
import sys
import os
from time import time
import yaml
import system_setup
import collections
import scipy.io

sys.path.append(os.path.expanduser("~/projects/shape_sharing/src/"))
from common import voxlets, scene, mesh


if len(sys.argv) > 1:
    parameters_path = sys.argv[1]
else:
    parameters_path = './testing_params_nyu.yaml'
parameters = yaml.load(open(parameters_path))

plot_gt_oracle = False

if parameters['testing_data'] == 'oisin_house':
    import real_data_paths as paths
elif parameters['testing_data'] == 'synthetic':
    import synthetic_paths as paths
elif parameters['testing_data'] == 'nyu_cad':
    import nyu_cad_paths as paths
else:
    raise Exception('Unknown training data')

params = [xx for xx in parameters['tests']
	if xx['name'] == 'short_samples_no_segment'][0]

def process_sequence(sequence):

    print "-> Loading ground truth", sequence['name']
    fpath = paths.prediction_folderpath % (parameters['batch_name'], sequence['name'])
    sys.stdout.flush()
    gt_scene = pickle.load(open(fpath + 'ground_truth.pkl'))
    results_dict = collections.OrderedDict()

    evaluation_region_loadpath = paths.evaluation_region_path % (
        parameters['batch_name'], sequence['name'])
        # pickle.dump(evaluation_region, open(savepath, 'w'), -1)
    evaluation_region = scipy.io.loadmat(
        evaluation_region_loadpath)['evaluation_region'] > 0

    for alpha in [0.0, 5, 10, 50, 100, 500, 1000, 10000]:

        print "Alpha is ", alpha

        prediction_savepath = fpath + params['name'] + ('_%f_alpha.pkl' % alpha)

        if os.path.exists(prediction_savepath):

            prediction = pickle.load(open(prediction_savepath))

            # sometimes multiple predictions are stored in predicton
            results_dict[alpha] = \
                gt_scene.evaluate_prediction(prediction.V,
                    voxels_to_evaluate=evaluation_region)

        else:
            print "Could not load ", prediction_savepath

    return results_dict


# need to import these *after* the pool helper has been defined
if system_setup.multicore:
    import multiprocessing
    mapper = multiprocessing.Pool(8).map
else:
    mapper = map


def get_mean_score(test, all_scores, score_type):
    all_this_scores = []
    for sc in all_scores:
        if test not in sc:
            return np.nan
        if score_type in sc[test]:
            all_this_scores.append(sc[test][score_type])

    return np.array(all_this_scores).mean()


if __name__ == '__main__':

    # print "WARNING - SMALL TEST DATA"
    # test_data = yaml.load(open('/media/ssd/data/oisin_house/train_test/test.yaml'))
    results = mapper(process_sequence, paths.test_data)
    yaml.dump(results, open('./nyu_cad/all_results.yaml', 'w'))

    # printing the accumulated table
    scores = ['iou', 'precision', 'recall']

    print '\n'
    print ' ' * 25,
    for score in scores:
        print score.ljust(10),
    print '\n' + '-' * 55

    sizes = []

    for experiment_name in results[0]:
        print str(experiment_name).ljust(25),
        for score_type in scores:
            score = get_mean_score(experiment_name, results, score_type)
            print ('%0.3f' % score).ljust(10),
        print ""
