import matplotlib.pyplot as plt
import numpy as np
import cPickle as pickle
import sys
import os
from time import time
import yaml
import real_data_paths as paths
import system_setup
import collections
sys.path.append(os.path.expanduser("~/projects/shape_sharing/src/"))
from common import voxlets, scene

parameters_path = './testing_params.yaml'
parameters = yaml.load(open(parameters_path))


def process_sequence(sequence):

    print "-> Loading ground truth", sequence['name']
    fpath = paths.prediction_folderpath % (parameters['batch_name'], sequence['name'])
    gt_scene = pickle.load(open(fpath + 'ground_truth.pkl'))

    results_dict = collections.OrderedDict()

    for test_params in parameters['tests']:

        if test_params['name'] == 'ground_truth':
            continue

        prediction_savepath = fpath + test_params['name'] + '.pkl'
        if os.path.exists(prediction_savepath):

            prediction = pickle.load(open(prediction_savepath))

            # sometimes multiple predictions are stored in predicton
            if hasattr(prediction, '__iter__'):
                for key, item in prediction.iteritems():
                    results_dict[test_params['name'] + str(key)] = \
                        gt_scene.evaluate_prediction(item.V)
            else:
                results_dict[test_params['name']] = \
                    gt_scene.evaluate_prediction(prediction.V)

            if test_params['name'] == 'ground_truth_oracle':
                diff = prediction.V - gt_scene.gt_tsdf.V
                plt.subplot(221)
                plt.imshow(gt_scene.voxels_to_evaluate.reshape(gt_scene.gt_tsdf.V.shape)[:, :, 20])
                plt.subplot(222)
                plt.imshow(gt_scene.gt_tsdf.V[:, :, 20], cmap=plt.get_cmap('bwr'))
                plt.subplot(223)
                plt.imshow(diff[:, :, 20], cmap=plt.get_cmap('bwr'))
                plt.clim(-0.02, 0.02)
                plt.colorbar()
                plt.subplot(224)
                plt.imshow(prediction.V[:, :, 20], cmap=plt.get_cmap('bwr'))

                gen_renderpath = paths.voxlet_prediction_img_path % \
                    (parameters['batch_name'], sequence['name'], '%s')
                plt.savefig(gen_renderpath % 'to_evaluate')

        else:
            print "Could not load ", prediction_savepath

    yaml_path = paths.scores_path % \
        (parameters['batch_name'], sequence['name'])

    with open(yaml_path, 'w') as f:
        f.write(yaml.dump(results_dict, default_flow_style=False))

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

    print "WARNING - SMALL TEST DATA"
    # test_data = yaml.load(open('/media/ssd/data/oisin_house/train_test/test.yaml'))
    results = mapper(process_sequence, paths.test_data)
    yaml.dump(results, open('./different_train_test/all_results.yaml', 'w'))

    # printing the accumulated table
    scores = ['iou', 'precision', 'recall']

    print '\n'
    print ' ' * 25,
    for score in scores:
        print score.ljust(10),
    print '\n' + '-' * 55

    for experiment_name in results[0]:
        print experiment_name.ljust(25),
        for score_type in scores:
            score = get_mean_score(experiment_name, results, score_type)
            print ('%0.4f' % score).ljust(10),
        print ""

            # results_dict[desc] = {
            #             'description': desc,
            #             'auc':         float(dic['auc']),
            #             'iou':         float(dic['iou']),
            #             'precision':   float(dic['precision']),
            #             'recall':      float(dic['recall'])}
