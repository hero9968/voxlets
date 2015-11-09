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
    # if sequence['name']

    # saving all these predictions to disk
    yaml_path = paths.scores_path % \
        (parameters['batch_name'], sequence['name'])

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

    for test_params in parameters['tests']:

        print test_params['name'],
        if test_params['name'] == 'ground_truth':
            continue

        prediction_savepath = fpath + test_params['name'] + '_keeping_existing.pkl'
        if os.path.exists(prediction_savepath):

            prediction = pickle.load(open(prediction_savepath))

            # sometimes multiple predictions are stored in predicton
            if hasattr(prediction, '__iter__'):
                for key, item in prediction.iteritems():
                    results_dict[test_params['name'] + str(key)] = \
                        gt_scene.evaluate_prediction(item.V,
                            voxels_to_evaluate=evaluation_region)
            else:
                results_dict[test_params['name']] = \
                    gt_scene.evaluate_prediction(prediction.V,
                        voxels_to_evaluate=evaluation_region)

            if test_params['name'] == 'ground_truth_oracle' and plot_gt_oracle:
                print "Rendering..."
                diff = prediction.V - gt_scene.gt_tsdf.V
                plt.subplot(221)
                plt.imshow(gt_scene.voxels_to_evaluate.reshape(
                    gt_scene.gt_tsdf.V.shape)[:, :, 20])
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

    # loading zheng predictions from the yaml files
    try:
        results_dict["zheng_2"] = yaml.load(open(
            paths.implicit_predictions_dir % ("zheng_2", sequence['name']) + 'eval.yaml'))
    except:
        results_dict["zheng_2"] = {
            'iou': np.nan, 'precision': np.nan, 'recall': np.nan}
    #
    # results_dict["zheng_3"] = yaml.load(open(
    #     paths.implicit_predictions_dir % ("zheng_3", sequence['name']) + 'eval.yaml'))


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

# FOR PLOTTING SIZE GRAPH:
# %matplotlib inline
# import numpy as np
# import matplotlib.pyplot as plt

# # sizes = [('0.005', 0.58920613885043516), ('0.002', 0.54050961991053581), ('0.004', 0.58708141996357788), ('0.0025', 0.57453766200014789), ('0.00125', 0.34895851355658664), ('0.01', 0.41102637929019542), ('0.0125', 0.2880756394244543)]
# sizes= [(0.005, 0.58920613885043516, 0.70984726997443048, 0.79551942011236931), (0.002, 0.54050961991053581, 0.8632621227991214, 0.59782704742789716), (0.004, 0.58708141996357788, 0.71263895650791054, 0.79324306790934684), (0.0025, 0.57453766200014789, 0.79758093552027287, 0.68863955851399294), (0.00125, 0.34895851355658664, 0.91452213982477382, 0.36130407836550488), (0.01, 0.41102637929019542, 0.78158512160913218, 0.47008189501931663), (0.0125, 0.2880756394244543, 0.80438995878420205, 0.31074675183359302)]

# # T  = [[s[0], s[1]] for s in sizes]
# TT = np.array(sizes)
# idxs = TT[:, 0].argsort()
# print idxs

# print TT

# plt.figure(figsize=(9, 6))
# plt.plot(30 * TT[idxs, 0] * 100, TT[idxs, 1], '-r', label='IoU')
# plt.plot(30 * TT[idxs, 0] * 100, TT[idxs, 2], '--g', label='Precision')
# plt.plot(30 * TT[idxs, 0] * 100, TT[idxs, 3], ':b', label='Recall')
# t = 15
# plt.plot([t, t], [0, 1], ':k', label='Voxlet size for other experiments')
# plt.xlabel('$x$ (cm). Voxlet is of size $x$ by $2x$ by $x$')
# plt.ylabel('IoU')
# plt.ylim(0, 1)
# plt.legend(loc='best')
# plt.savefig('/home/michael/Desktop/vox_sizes.eps')
