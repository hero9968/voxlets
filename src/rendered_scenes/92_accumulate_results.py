# This will accumulate all the different results from all the different experiments and average them

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

tests = ['full_oracle_voxlets',
         'oracle_voxlets',
         'nn_oracle_voxlets',
         'greedy_oracle_voxlets',
         'pred_voxlets',
         'implicit']

all_scores = all_scores[1:5]
import numpy as np
for test in tests:
    mean_auc = np.array([sc[test]['auc'] for sc in all_scores]).mean()
    mean_precision = np.array([sc[test]['precision'] for sc in all_scores]).mean()
    mean_recall = np.array([sc[test]['recall'] for sc in all_scores]).mean()
    print '%s & %0.3f & %0.3f & %0.3f' % (test, mean_precision, mean_recall, mean_auc)
