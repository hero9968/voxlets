'''
a script to run on troll which will extract the data about the forest and save to disk
'''
import numpy as np
#import matplotlib.pyplot as plt 
import cPickle as pickle
import sys, os
sys.path.append(os.path.expanduser("~/projects/shape_sharing/src/"))

from common import paths


print "Loading forest"
forest_pca = pickle.load(open(paths.voxlet_model_pca_path, 'rb'))

print "getting features"
oob_score = forest_pca.oob_score_
importances = forest_pca.feature_importances_

D = dict(oob_score=oob_score, importances=importances)
import scipy.io 
scipy.io.savemat('./forest_data.mat', D)

