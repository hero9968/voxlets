import numpy as np
import scipy.io
from time import time
import yaml
import sys, os
import cPickle as pickle
sys.path.append(os.path.expanduser('~/projects/shape_sharing/src/'))
sys.path.append(os.path.expanduser('~/projects/shape_sharing/src/implicit/'))

from common import scene, voxel_data
import real_data_paths as paths

parameters = yaml.load(open('../implicit_params.yaml'))

the_zheng_parameter = 2
modelname = 'zheng_' + str(the_zheng_parameter)

for sequence in paths.test_data:

    sequence['folder'] = '/media/michael/Seagate/phd_projects/volume_completion_data/data/oisin_house/' + \
        sequence['folder'].split('/')[-2] + '/'
    print sequence


    print "Processing " + sequence['name']
    sc = scene.Scene(parameters['mu'], None)
    sc.load_sequence(
        sequence,
        segment_base=0.03,
        frame_nos=0,
        segment_with_gt=True,
        segment=True,
        save_grids=False)

    # loading the predcitoin
    P = '/media/michael/Seagate/phd_projects/volume_completion_data/data/oisin_house/implicit/models/zheng_2/predictions/'
    pred_grid = pickle.load(open(P + sequence['name'] + '/prediction.pkl'))

    pred_grid2 = pred_grid.copy()
    pred_grid2.V[sc.im_tsdf.V > 0] = np.nanmax(pred_grid2.V)

    # doing thr render
    pred_grid.render_view(P + sequence['name'] + '/prediction_render.png',
        xy_centre=True, ground_height=0.03, keep_obj=True, actually_render=False)
    pred_grid2.render_view(P + sequence['name'] + '/prediction_render2.png',
        xy_centre=True, ground_height=0.03, keep_obj=True, actually_render=False)
