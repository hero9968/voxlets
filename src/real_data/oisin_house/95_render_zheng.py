
import numpy as np
import cPickle as pickle
import sys
import os
sys.path.append(os.path.expanduser("~/projects/shape_sharing/src/"))
from common import voxlets, scene


to_render = [1,32,195,564,591,620,520,698]

op_dir = '/home/michael/Desktop/baselines_voxelised/'
ip_dir = "/home/michael/projects/shape_sharing/data/cleaned_3D/renders_yaml_format/silberman_split/implicit/models/zheng_2_real/predictions/"

mapper = {}
for xx in os.listdir(ip_dir):
	mapper[int(xx.split('_')[0])] = xx


def process_sequence(idx):

    print "-> Loading ground truth", idx

    fpath = ip_dir + mapper[idx] + '/prediction.pkl'
	
    zheng = pickle.load(open(fpath))

    zheng.render_view(op_dir + str(idx) + '_zheng.png',
        xy_centre=False, ground_height=0,
        keep_obj=True, actually_render=False,
        flip=True)


for idx in to_render:
	process_sequence(idx)
