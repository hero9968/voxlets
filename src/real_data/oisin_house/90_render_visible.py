
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
        save_grids=False, load_implicit=False, voxel_normals='gt_tsdf')
    print sc.gt_tsdf.origin

    # Path where any renders will be saved to
    gen_renderpath = paths.voxlet_prediction_img_path % \
        (test_type, sc.sequence['name'], '%s')

    sc.im_tsdf.render_view(gen_renderpath % 'visible', xy_centre=True, ground_height=None)



