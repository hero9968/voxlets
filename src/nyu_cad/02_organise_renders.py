# put the NYU cad files in a standard format

import os, sys, shutil
import binvox_rw
import cPickle as pickle
import yaml
import scipy.io
sys.path.append('../../src/')
from common import voxel_data
import numpy as np
import skimage.io
from scipy.ndimage.interpolation import zoom

base_dir = '../../data/cleaned_3D/'

this_nowalls_dir = base_dir + 'renders_no_walls/'
this_walls_dir = base_dir + 'renders_with_walls/'
new_dir = base_dir + 'renders_yaml_format/renders/'

for foldername in os.listdir(this_walls_dir):

    print "Doing ", foldername

    # make folder
    if not os.path.exists(new_dir + foldername):
        os.makedirs(new_dir + foldername)

    new_path = new_dir + foldername + '/images'
    if not os.path.exists(new_path):
        os.makedirs(new_path)
    else:
        print "\tSeems we've already done this one... SKIPPING"
        continue

    print new_path

    # move images
    for imname in ['/depth.png', '/rgb.png']:
        im_path = this_walls_dir + foldername + imname
        print im_path
        shutil.copy(im_path, new_path + imname)

    # maybe now try to create a mask...
    no_walls_loadpath = this_nowalls_dir + foldername + '/depth.png'
    no_walls_depth = skimage.io.imread(no_walls_loadpath)
    mask = (no_walls_depth != 2**16 -1).astype(np.uint8) * 255

    skimage.io.imsave(new_path + '/mask.png', mask)

    # now load the binvox and save as a pickle file in the correct place
    bvox_path = this_nowalls_dir + foldername + '.binvox'
    if not os.path.exists(bvox_path):
        print "\tNot done binvox yet! SKIPPING"
        continue

    with open(bvox_path, 'r') as f:
        bvox = binvox_rw.read_as_3d_array(f)

    print bvox.data.shape
    print bvox.translate
    print bvox.scale
    print bvox.axis_order

    vgrid = voxel_data.WorldVoxels()
    vgrid.set_origin(bvox.translate)
    vgrid.set_voxel_size(bvox.scale / float(bvox.data.shape[0]))
    vgrid.V = bvox.data.astype(np.float16)
    vgrid.V = vgrid.compute_tsdf(0.1)

    tempV = vgrid.V.copy().transpose((0, 2, 1))[:, :, :]
    vgrid.V = tempV
    vgrid.origin = vgrid.origin[[0, 2, 1]]

    with open(new_dir + foldername + '/ground_truth_tsdf.pkl', 'w') as f:
        pickle.dump(vgrid, f, -1)

    # loading the camera rotation matrix
    mat = scipy.io.loadmat(base_dir + '/mat/%s.mat' % foldername)
    K = mat['model'][0]['camera'][0]['K'][0, 0]
    R = mat['model'][0]['camera'][0]['R'][0, 0]
    R[1, :] *= -1
    R[2, :] *= -1

    temp = R[1, :].copy()
    R[1, :] = R[2, :]
    R[2, :] = temp

    R2 = np.eye(4)
    R2[:3, :3] = R


    K = np.array([[ 570.,    0.,  320.],
                  [   0.,  570.,  240.],
                  [   0.,    0.,    1.]])

    # finally make the yaml file
    poses = [{
        'camera': 1,
        'depth_scaling': 4.0,
        'frame': 0,
        'id': '01_0000',
        'image': 'images/depth.png',
        'rgb': 'images/rgb.png',
        'mask': 'images/mask.png',
        'intrinsics': K.ravel().tolist(),
        'pose': R2.ravel().tolist(),
        'timestamp': 0.0
    }]
    yaml_path = new_dir + foldername + '/poses.yaml'
    print yaml_path
    yaml.dump(poses, open(yaml_path, 'w'))
