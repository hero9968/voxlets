'''
Need the labelling!
'''

import sys
import os
import scipy.io
from time import time
import yaml
import numpy as np
sys.path.append(os.path.expanduser('~/projects/shape_sharing/src/'))
import cPickle as pickle
import system_setup
import paths
from common import scene, features, corr_clust

parameters = yaml.load(open('./params.yaml'))

labels = yaml.load(open('/media/ssd/data/oisin_house/obj_discovery/labelling/labels.yaml'))

print "Creating output folder"
savefolder = paths.models_dir
if not os.path.exists(savefolder):
    os.makedirs(savefolder)

# loading in all the data
all_X = []
all_Y = []

for sequence in paths.all_train_data:

    load_location = paths.features_dir + sequence['name'] + '.mat'
    D = pickle.load(open(load_location))
    print D['features'].keys()

    # load in the features and labels
    for label, region_feature_dict in D['features'].items():
        all_X.append(features.combine_features(region_feature_dict))
        print labels[sequence['scene']].keys()
        all_Y.append(labels[sequence['scene']][int(label)])


all_X_np = np.array(all_X)
all_Y_np = np.array(all_Y)

print "Final shapes are "
print all_X_np.shape, all_Y_np.shape
print all_X_np.dtype, all_Y_np.dtype
print np.isnan(all_X_np).sum(), np.isnan(all_Y_np).sum()

cc = corr_clust.CorrClust()
# cc.train(
