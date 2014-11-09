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

############################################
"PARAMTERS"
thresholds = np.linspace(0, 1, 100)
pred_types = ['oma', 'modal', 'medioid', 'bb']#, 'bpc', 'no_spider']

############################################
"Setting up the dictionary to store results"
metrics = {}
for pred_type in pred_types:
    metrics[pred_type] = {}
    metrics[pred_type]['tpr'] = []
    metrics[pred_type]['fpr'] = []

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

            print "loading the data"
            D = scipy.io.loadmat(loadpath)

            print "Converting"
            met = voxel_data.VoxMetrics()
            met.set_gt(D['gt'])
        
            met.set_pred(D['prediction'])
            fpr, tpr = met.compute_tpr_fpr(thresholds)

            metrics[pred_type]['tpr'].append(tpr)
            metrics[pred_type]['fpr'].append(fpr)

        print "Done " + modelname

############################################
"SAVING"
import scipy.io
scipy.io.savemat('roc_curve_data.mat', metrics)
