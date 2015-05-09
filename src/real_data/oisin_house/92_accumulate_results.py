# This will accumulate all the different results from all the different experiments and average them
import numpy as np
import sys
import os
sys.path.append(os.path.expanduser("~/projects/shape_sharing/src/"))
import yaml
import real_data_paths as paths
# Choose nearest I think has just one mask for each prediction
# test_types = ['different_data_split', 'different_data_split_dw_empty', 'different_data_split_deep_dw_empty']
test_types = ['different_data_split', 'different_data_split_dw_empty', 'different_data_split_deep_dw_empty',
'depth_experiments/10/', 'depth_experiments/15/', 'depth_experiments/20/', 'depth_experiments/25/']
# test_types = ['oma', 'oma_masks', 'oma_choose_nearest', 'oma_choose_nearest_average_mask', 'oma_medioid']

def get_all_scores(test_type):
    all_scores = []

    for sequence in paths.test_data:

        # print sequence['name']

        fpath = paths.voxlet_prediction_folderpath % \
            (test_type, sequence['name'])

        if os.path.exists(fpath + 'scores.yaml'):
            with open(fpath + 'scores.yaml', 'r') as f:
                all_scores.append(yaml.load(f))
        else:
            print "Not found"
            pass
    return all_scores


def get_mean_score(test, all_scores, score_type):
    all_this_scores = []
    for sc in all_scores:
        if test not in sc:
            return np.nan
        if score_type in sc[test]:
            all_this_scores.append(sc[test][score_type])
    # print score_type
    # print np.array(all_this_scores)
    # print np.array(all_this_scores).mean()

    # print "Score for %s, %s is length %d" % (test, score_type, len(all_this_scores))
    # print score_type
    # if score_type=='precision':
    #     print np.sort(all_this_scores)

    return np.array(all_this_scores).mean()

tests = ['pred_voxlets']
# , 'Name', 'Medioid', 'pred_remove_excess']
# , 'pred_voxlets', 'pred_voxlets_exisiting', 'pred_remove_excess']

# print test_type

print '\t prec \t rec \t  iou\n'
for test_type in test_types:

    print test_type

    all_scores = get_all_scores(test_type)
    for test in tests:

        # mean_auc = get_mean_score(test, all_scores, 'auc')
        mean_precision = get_mean_score(test, all_scores, 'precision')
        mean_recall = get_mean_score(test, all_scores, 'recall')
        mean_iou = get_mean_score(test, all_scores, 'iou')

        print '%s \t %0.3f \t %0.3f \t %0.3f \\\\' % (test[:5], mean_precision, mean_recall, mean_iou)
