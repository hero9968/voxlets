'''
a script to load in a load of voxelised results, and to extract roc data for each one
prob can't load all at once so will instead get ROC curves for each one
'''

import numpy as np
import matplotlib.pyplot as plt 
import scipy.io
import sys, os
sys.path.append(os.path.expanduser("~/projects/shape_sharing/src/"))

from common import paths
from common import voxel_data
import sklearn.metrics
############################################
"PARAMTERS"
pred_types = ['oma', 'modal', 'just_spider', 'just_cobweb']#, 'bpc', 'no_spider']

############################################
"Setting up the dictionary to store results"
metrics = {}
for pred_type in pred_types:
    metrics[pred_type] = {}
    metrics[pred_type]['auc'] = []
    metrics[pred_type]['prescision'] = []
    metrics[pred_type]['recall'] = []

############################################
"MAIN LOOP"

for pred_type in pred_types:

    print "Doing pred type  " + pred_type
    # loop over each output saved file
    for modelname in paths.test_names:
        print "Doing model " + modelname
        for this_view_idx in [0, 10, 20, 30, 40]:

            test_view = paths.views[this_view_idx]

            # TODO - will eventually make this so each prediciton type is in a different folder or something
            if paths.host_name == 'troll':
                loadpath = paths.base_path + '/voxlets/bigbird/predictions/%s/%s_%s.mat' % (pred_type, modelname, test_view)
            else:
                loadpath = paths.base_path + '/voxlets/bigbird/troll_predictions/%s/%s_%s.mat' % (pred_type, modelname, test_view)

            "loading the data"
            D = scipy.io.loadmat(loadpath)

            "Converting"
            bin_gt = ((D['gt'].flatten() + 0.03) / 0.06).astype(int)
            pred_scaled = ((D['prediction'].flatten() + 0.03) / 0.06).astype(float)
            prescision = sklearn.metrics.precision_score(bin_gt, pred_scaled.astype(int))
            recall = sklearn.metrics.recall_score(bin_gt, pred_scaled.astype(int))
            auc = sklearn.metrics.roc_auc_score(bin_gt, D['prediction'].flatten())

            metrics[pred_type]['auc'].append(auc)
            metrics[pred_type]['prescision'].append(prescision)
            metrics[pred_type]['recall'].append(recall)

        print "Done " + modelname

    print "------------------"
    print pred_type
    print np.mean(metrics[pred_type]['auc'])
    print np.mean(metrics[pred_type]['prescision'])
    print np.mean(metrics[pred_type]['recall'])

############################################
"SAVING"
import scipy.io
if not os.path.exists('./data/'): os.makedirs('data')
scipy.io.savemat('data/roc_curve_data.mat', metrics)
