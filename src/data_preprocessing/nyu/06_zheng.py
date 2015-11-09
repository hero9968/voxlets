'''
make predictions for all of bigbird dataset
using my algorhtm
saves each prediction to disk
'''
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cPickle as pickle
import sys
import os
sys.path.append(os.path.expanduser("~/projects/shape_sharing/src/"))
from time import time
import scipy.misc  # for image saving
from scipy.io import loadmat, savemat
import shutil
import collections

import real_data_paths as paths
import real_params as parameters

from common import voxlets
from common import scene

import sklearn.metrics

# sys.path.append(os.path.expanduser("~/projects/shape_sharing/src/implicit/zheng"))

import implicit.zheng.find_axes as find_axes
the_zheng_parameter = 2

def process_sequence(sequence):
    sc = scene.Scene(parameters.mu, [])
    sc.load_sequence(sequence, frame_nos=0, segment_with_gt=False,
        save_grids=False, voxel_normals='im_tsdf')

    # loading the labels
    print sequence['scene']
    loadpath = '/media/ssd/data/nyu/data/nyu_labels/' + sequence['scene'] + '.mat'
    sc.gt_im_label = loadmat(loadpath)['labels']

    print "Doing zheng"
    pred_grid = find_axes.process_scene(sc, the_zheng_parameter)

    pred_grid.V = pred_grid.V.astype(np.float32)

    print "Converting to a tsdf equivalent"
    pred_grid.V[pred_grid.V > 0] = -1
    pred_grid.V[pred_grid.V == 0] = 1

    print (pred_grid.V < 0).sum()
    print (pred_grid.V > 0).sum()
    print (pred_grid.V == 0).sum()



    test_type = 'oma'


if __name__ == '__main__':

    for s in paths.test_data:
        print s['scene']
    temp = [s for s in paths.test_data if s['name'] == '0416_[0]']
    print temp
    tic = time()

    # need to import these *after* the pool helper has been defined
    if False:
        import multiprocessing
        multiprocessing.Pool(6).map(process_sequence, temp)
    else:
        map(process_sequence, temp)

    print "In total took %f s" % (time() - tic)
