# This will accumulate all the different results from all the different experiments and average them
import numpy as np
import sys
import os
sys.path.append(os.path.expanduser("~/projects/shape_sharing/src/"))
import yaml
from common import paths

test_type = 'oma_implicit'
all_scores = []

for sequence in paths.RenderedData.test_sequence():

    fpath = paths.RenderedData.voxlet_prediction_folderpath % \
        (test_type, sequence['name'])

    if os.path.exists(fpath + 'scores.yaml'):
        with open(fpath + 'scores.yaml', 'r') as f:
            all_scores.append(yaml.load(f))

def get_mean_score(test, score_type):
    all_this_scores = []
    for sc in all_scores:
        if score_type in sc[test]:
            all_this_scores.append(sc[test][score_type])
    return np.array(all_this_scores).mean()

tests = ['full_oracle_voxlets',
         'oracle_voxlets',
         'nn_oracle_voxlets',
         'greedy_oracle_voxlets',
         'pred_voxlets',
         'implicit']

all_scores = all_scores[1:5]
for test in tests:

    mean_auc = get_mean_score(test, 'auc')
    mean_precision = get_mean_score(test, 'precision')
    mean_recall = get_mean_score(test, 'recall')

    print '%s & %0.3f & %0.3f & %0.3f' % (test, mean_precision, mean_recall, mean_auc)
