
import numpy as np
import cPickle as pickle
import sys
import os
import yaml
import system_setup
import scipy.misc
import shutil
import matplotlib.pyplot as plt
sys.path.append(os.path.expanduser("~/projects/shape_sharing/src/"))
from common import voxlets, scene, mesh
import subprocess as sp

# ip_dir = "/home/michael/prism/data5/projects/depth_completion/cvpr2016/tabletop/"
ip_dir = ("/media/michael/Seagate/phd_projects/volume_completion_data/"
    "data/oisin_house/predictions/cvpr2016/")

zheng_ip_dir = ('/media/michael/Seagate/phd_projects/volume_completion_data/'
    'data/oisin_house/implicit/models/zheng_2/predictions/')

# op_dir = "/home/michael/Desktop/renders/"
op_dir = "/home/michael/projects/shape_sharing/media/cvpr2016_sup/imgs/tabletop_renders/"

tests = ['ground_truth']  # ['input', 'input_depth']#'ground_truth', 'short_and_tall_samples_no_segment', 'visible']


print "-> Main renders"
for sequence in os.listdir(ip_dir):

    print "-> Sequence: ", sequence

    for test in tests:
        print "-> Test: ", test

        # create op_dir
        savedir = op_dir + sequence + '/'

        if not os.path.exists(savedir):
            os.makedirs(savedir)

        if test == 'input' or test == 'input_depth':
            obj_path = ip_dir + sequence + '/' + test + '.png'
            shutil.copy(obj_path, savedir)
            continue

        # load obj
        if test == 'zheng':
            obj_path = zheng_ip_dir + sequence + '/' + 'prediction_render.png'
            shutil.copy(obj_path + '.obj', savedir + 'zheng.png.obj')
        else:
            obj_path = ip_dir + sequence + '/' + test + '.png'
            shutil.copy(obj_path + '.obj', savedir)

        if test == 'ground_truth':
            ms = mesh.Mesh()
            ms.load_from_obj(obj_path + '.obj')
            ms.vertices[:, 0] *= -1
            ms.write_to_obj(savedir + '/' + test + '.png.obj')

        # do the render
        sys.stdout.flush()
        blend_path = os.path.expanduser(
            '~/projects/shape_sharing/src/rendered_scenes/spinaround/spin_closer.blend')
        blend_py_path = os.path.expanduser(
            '~/projects/shape_sharing/src/rendered_scenes/spinaround/blender_spinaround_frame.py')
        subenv = os.environ.copy()
        subenv['BLENDERSAVEFILE'] = savedir + test + '.png'
        sp.call(['blender',
                 blend_path,
                 "-b", "-P",
                 blend_py_path],
                 env=subenv,
                 stdout=open(os.devnull, 'w'),
                 close_fds=True)

        # shutil.copy(ip_dir + sequence + '/' + test + '.png.obj', savedir + )
