
import numpy as np
import cPickle as pickle
import sys
import os
import yaml
import system_setup
import scipy.misc
import matplotlib.pyplot as plt
sys.path.append(os.path.expanduser("~/projects/shape_sharing/src/"))
from common import voxlets, scene, mesh

parameters_path = './testing_params_nyu_real.yaml'
import nyu_cad_paths_silberman as paths

to_render = [1,32,195,564,591,620,520,698]

# op_dir = '/home/michael/Desktop/baselines_voxelised/'
op_dir = '/home/michael/prism/data5/projects/depth_completion/cvpr2016/nyu/from_cad/'
ip_dir = "/media/michael/Seagate/phd_projects/volume_completion_data/data/nyu_cad/predictions/nyu_cad_silberman_original/"

mapper = {}
for xx in os.listdir(ip_dir):
    mapper[int(xx.split('_')[0])] = xx

def process_sequence(idx):


    if not os.path.exists(op_dir + mapper[idx]):
        os.makedirs(op_dir + mapper[idx])

    print "-> Loading ground truth", idx

    fpath = ip_dir + mapper[idx] + '/pickles/ground_truth.pkl'
    gt_scene = pickle.load(open(fpath))

    print "-> Loading prediction", idx
    #
    # fpath = ip_dir + mapper[idx] + '/short_tall_samples_0.02_pointwise.pkl'
    # prediction = pickle.load(open(fpath))


    # rendering visible one way
    print "-> Rendering visible 1", idx
    savepath = op_dir + mapper[idx] + "/visible1.png"
    vis_ms = gt_scene.render_visible(
        savepath, xy_centre=False, keep_obj=True, actually_render=False,
        flip=True)
    # vis_ms.vertices[:, 0] *= -1

    # load obj
    # ms = mesh.Mesh()
    # ms.load_from_obj(ip_dir + mapper[idx] + '/short_tall_samples_0.025_pointwise.png.obj')
    #
    # print "-> Rendering ours with visible...", idx
    # N = vis_ms.vertices.shape[0]
    # vis_ms.vertices = np.vstack((vis_ms.vertices, ms.vertices))
    # vis_ms.faces = np.vstack((vis_ms.faces, ms.faces + N))
    # vis_ms.write_to_obj(op_dir + mapper[idx] + '/ours_with_visible.obj')

    print "-> Rendering zheng with visible...", idx
    ms = mesh.Mesh()
    ms.load_from_obj(op_dir + mapper[idx] + '/zheng_real.png.obj')

    N = vis_ms.vertices.shape[0]
    vis_ms.vertices = np.vstack((vis_ms.vertices, ms.vertices))
    vis_ms.faces = np.vstack((vis_ms.faces, ms.faces + N))
    vis_ms.write_to_obj(op_dir + mapper[idx] + '/zheng_real_with_visible.obj')

    # rendering visible another way
    print "-> Rendering visible 2", idx
    savepath = op_dir + mapper[idx] + "visible2.png"
    gt_scene.im_tsdf.render_view(
        savepath,
        xy_centre=False, ground_height=0, flip=True,
        keep_obj=True, actually_render=False)

import shutil

def prc2(idx):
    for nm in ['lin_voxelised.png.obj', 'gcpr_voxelised.png.obj', 'visible2.png.obj', 'zheng.png.obj']:
        print idx, nm
        ip = '/home/michael/Desktop/baselines_voxelised/' + str(idx) + \
            '_' + nm
        if 'zheng' in nm:
            nm = 'zheng_real.png.obj'
        op = op_dir + mapper[idx] + '/' + nm

        shutil.copy(ip, op)

for idx in to_render:
    process_sequence(idx)
