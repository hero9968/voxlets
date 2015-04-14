# This will accumulate all the different results from all the different experiments and average them
import numpy as np
import sys
import os
sys.path.append(os.path.expanduser("~/projects/shape_sharing/src/"))
import yaml
import paths

test_types = ['oma', 'oma_no_masks', 'oma_with_weights_broke_maybe']

def get_all_scores(test_type):
    all_scores = []

    for sequence in paths.RenderedData.test_sequence()[1:16]:

        # print sequence['name']

        fpath = paths.RenderedData.voxlet_prediction_folderpath % \
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
        if score_type in sc[test]:
            all_this_scores.append(sc[test][score_type])
    # print np.array(all_this_scores)
    # print "Score for %s, %s is length %d" % (test, score_type, len(all_this_scores))
    return np.array(all_this_scores).mean()

tests = ['OR1',
         'OR2',
         'OR3',
         'OR4',
         'pred_voxlets']

for test_type in test_types:
    print test_type
    all_scores = get_all_scores(test_type)
    for test in tests:

        mean_auc = get_mean_score(test, all_scores, 'auc')
        mean_precision = get_mean_score(test, all_scores, 'precision')
        mean_recall = get_mean_score(test, all_scores, 'recall')

        print '%s & %0.3f & %0.3f & %0.3f \\\\' % (test, mean_precision, mean_recall, mean_auc)
