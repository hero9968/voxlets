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
elif parameters['testing_data'] == 'nyu_cad_silberman':
    import nyu_cad_paths_silberman as paths
else:
    raise Exception('Unknown training data')


def process_sequence(sequence):

    fpath = paths.prediction_folderpath % (parameters['batch_name'], sequence['name'])
    results_dict = collections.OrderedDict()

    for test_params in parameters['tests']:
        if test_params['name'] == 'ground_truth' or test_params['name'] == 'visible':
            continue

        results_savepath = fpath + "../eval_%s.yaml" % test_params['name']
        if not os.path.exists(results_savepath):
            print "Not fiund ", results_savepath
            results_dict[test_params['name']] = {
                'iou': np.nan, 'precision': np.nan, 'recall': np.nan}
        else:

            results = yaml.load(open(results_savepath))
            results_dict[test_params['name']] = results

    # loading zheng predictions from the yaml files
    try:
        if parameters['original_nyu']:
            zheng_name = "zheng_2_real"
        else:
            zheng_name = "zheng_2"

        results_dict["zheng_2"] = yaml.load(open(
            paths.implicit_predictions_dir % (zheng_name, sequence['name']) + 'eval.yaml'))
    except:
        results_dict["zheng_2"] = {
            'iou': np.nan, 'precision': np.nan, 'recall': np.nan}

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

    return np.nanmean(np.array(all_this_scores))



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
        print experiment_name.ljust(25),
        for score_type in scores:
            score = get_mean_score(experiment_name, results, score_type)
            print ('%0.3f' % score).ljust(10),
        print ""

        # finding best and worst scores for this experiment
        all_ious = [result[experiment_name]['iou'] for result in results]
        all_ious = np.array(all_ious)
        iou_idxs = np.argsort(all_ious)[::-1]

        # now printing these best and worst results to screen...
        print "\n\tRank \t IOU   \t us/zheng_2 \t Name "
        print "\t" + "-" * 40
        for count, val in enumerate(iou_idxs):
            if 1:#count == all_ious.shape[0] // 2 or count < 5 or count > len(all_ious) - 5:
            # scene_name = results[val]['name']
                scene_name = paths.test_data[val]['name']
                this_iou = all_ious[val]
                us_v_zheng_2 = this_iou / results[val]['zheng_2']['iou']
                print "\t%d \t %0.3f \t %0.3f \t %s" % (count, this_iou, us_v_zheng_2, scene_name)
        print "\n"

        # this is not about sizes...
        if experiment_name.startswith('short'):
            iou = get_mean_score(experiment_name, results, 'iou')
            prec = get_mean_score(experiment_name, results, 'precision')
            rec = get_mean_score(experiment_name, results, 'recall')
            # sizes.append((float(experiment_name.split('_')[2]), iou, prec, rec))

    mapper = {
        'nn_oracle', '\\textbf\{V\}$_\\text\{nn\}$',
        'gt_oracle', '\\textbf\{V\}$_\\text\{gt\}$',
        'pca_oracle', '\\textbf\{V\}$_\\text\{pca\}$',
        'greedy_oracle', '\\textbf\{V\}$_\\text\{agg\}$',
        'medioid', 'Medoid',
        'mean', 'Mean',
        'short_and_tall_samples_no_segment', 'Ours',
        'bounding_box', 'Bounding box'}

    for experiment_name in results[0]:
        if experiment_name in mapper:
            print mapper[experiment_name].ljust(25) + " & "
        else:
            print experiment_name.ljust(25) + " & "
        for score_type in scores:
            score = get_mean_score(experiment_name, results, score_type)
            print ('%0.3f' % score).ljust(10),
            if score_type == 'recall':
                print "\\\\"
            else:
                print " & ",





    print "evaluate_inside_room_only is", parameters['evaluate_inside_room_only']
    print sizes

            # results_dict[desc] = {
            #             'description': desc,
            #             'auc':         float(dic['auc']),
            #             'iou':         float(dic['iou']),
            #             'precision':   float(dic['precision']),
            #             'recall':      float(dic['recall'])}
