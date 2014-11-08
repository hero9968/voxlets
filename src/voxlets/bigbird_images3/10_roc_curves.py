'''
a script to load in a load of voxelised results, and to extract roc data for each one
prob can't load all at once so will instead get ROC curves for each one
'''

import numpy as np
import matplotlib.pyplot as plt 
import cPickle as pickle
import sys, os
sys.path.append(os.path.expanduser("~/projects/shape_sharing/src/"))
#import sklearn.metrics

from common import paths

thresholds = np.linspace(0, 1, 100)

def convert_to_flat_in_zero_one(V_in):
    return 1 - (V_in.flatten() + 0.03) / 0.06

all_modal_tpr = []
all_modal_fpr = []
all_med_tpr = []
all_med_fpr = []


# loop over each output saved file
for modelname in paths.test_names:
    print "Doing model " + modelname
    for this_view_idx in [0, 10, 20, 30, 40]:

        test_view = paths.views[this_view_idx]
        print "Doing view " + test_view

        loadpath = '/Users/Michael/projects/shape_sharing/data/voxlets/bigbird/troll_predictions/%s_%s.pkl' % (modelname, test_view)

        print "loading the data"
        f = open(loadpath, 'rb')
        D = pickle.load(f)
        f.close()

        print "Converting"
        med = convert_to_flat_in_zero_one(D['medioid'])
        modal = convert_to_flat_in_zero_one(D['modal'])
        GT = convert_to_flat_in_zero_one(D['gt'])

        N = GT.shape[0]

        print "Doing analysis"
        temp_med_fpr = []
        temp_med_tpr = []
        temp_modal_tpr = []
        temp_modal_fpr = []
        for thres in thresholds:

            fp = np.sum(np.logical_and(med>thres, GT<0.5))
            temp_med_fpr.append(float(fp)/float(N))

            print fp 

            fp = np.sum(np.logical_and(modal>thres, GT<0.5))
            temp_modal_fpr.append(float(fp)/float(N))

            print fp 

            tp = np.sum(np.logical_and(med>thres, GT>=0.5))
            temp_med_tpr.append(float(tp)/float(N))

            print tp

            tp = np.sum(np.logical_and(modal>thres, GT>=0.5))
            temp_modal_tpr.append(float(tp)/float(N))

            print tp
            

        all_modal_tpr.append(temp_modal_tpr)
        all_modal_fpr.append(temp_modal_fpr)
        all_med_tpr.append(temp_modal_tpr)
        all_med_fpr.append(temp_modal_fpr)

    print "Done " + modelname


# convert them to np arrays
np_all_modal_tpr = np.array(all_modal_tpr)
np_all_modal_fpr = np.array(all_modal_fpr)
np_all_med_tpr = np.array(all_med_tpr)
np_all_med_fpr = np.array(all_med_fpr)

print np_all_modal_tpr.shape
print np_all_modal_fpr.shape
print np_all_med_tpr.shape
print np_all_med_fpr.shape

# now save them all to disk
D = dict(np_all_modal_tpr=np_all_modal_tpr, 
         np_all_modal_fpr=np_all_modal_fpr, 
         np_all_med_tpr=np_all_med_tpr, 
         np_all_med_fpr=np_all_med_fpr)

import scipy.io
scipy.io.savemat('roc_curve_data.mat', D)
