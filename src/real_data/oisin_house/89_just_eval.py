
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
import scipy.io
import yaml
import shutil
import collections

import real_data_paths as paths
import real_params as parameters

from common import voxlets
from common import scene

import sklearn.metrics


test_type = 'mixed_voxlets2'


for sequence in paths.test_data:


    print "Processing ", sequence['name']
    sc = scene.Scene(parameters.mu, [])
    sc.load_sequence(sequence, frame_nos=0, segment_with_gt=True,
        save_grids=False, load_implicit=False, voxel_normals='gt_tsdf', carve=True)
    print sc.gt_tsdf.origin

    gen_renderpath = paths.voxlet_prediction_img_path % \
        (test_type, sequence['name'], '%s')

    results_dict = {}
    for desc in ['pred_remove_excess', 'Name', 'Medioid']:

        D = scipy.io.loadmat(gen_renderpath.replace('png', 'mat') % desc)

        dic = sc.evaluate_prediction(D['grid'])
        print dic
        results_dict[desc] = \
                    {'description': desc,
                     'auc':         float(dic['auc']),
                     'iou':         float(dic['iou']),
                     'precision':   float(dic['precision']),
                     'recall':      float(dic['recall'])}

    # print results_dict

    fpath = paths.voxlet_prediction_folderpath % \
        (test_type, sequence['name'])

    with open(fpath + 'scores.yaml', 'w') as f:
        f.write(yaml.dump(results_dict, default_flow_style=False))

