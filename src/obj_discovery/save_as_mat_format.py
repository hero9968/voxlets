'''
Aim is to load the data from the python format and then to resave in the format
required by my matlab code
'''

import sys
import os
import scipy.io
from scipy.misc import imsave
from time import time
import yaml
import numpy as np
sys.path.append(os.path.expanduser('~/projects/shape_sharing/src/'))
sys.path.append(os.path.expanduser('~/projects/shape_sharing/src/real_data/'))

import matplotlib.pyplot as plt
from oisin_house import real_data_paths as paths

from common import scene, carving, features

print "Creating output folder"
savefolder = '/media/ssd/data/oisin_house/for_object_discovery/'
if not os.path.exists(savefolder):
    os.makedirs(savefolder)

for idx, sequence in enumerate(paths.test_data):

    print "Processing " + sequence['name']
    sc = scene.Scene(0.025, None)
    sc.load_sequence(sequence, frame_nos=0, segment_with_gt=False,
        segment=False, save_grids=False, carve=True, voxel_normals='im_tsdf')

    # creating a new mask ignoring points on the plane...
    heights = sc.im.get_world_xyz()[:, 2].reshape(sc.im.depth.shape)
    not_in_floor = heights > 0.05
    mask = np.logical_and(sc.im.mask, not_in_floor)

    # compting normals
    ne = features.Normals()
    norms = ne.compute_bilateral_normals(sc.im)

    # # saving the depth and the mask together
    # imsave(savefolder + sequence['name'] + '_mask.png', mask)
    # imsave(savefolder + sequence['name'] + '_rgb.png', sc.im.rgb)

    RR = {}
    RR['R'] = sc.im.cam.H[:3, :3]
    RR['points3d'] = sc.im.reproject_3d().T
    RR['normals'] = norms

    D = dict(
        mask=mask.astype(np.uint8),
        imgDepth=sc.im.depth,
        seqname=sequence['name'])

    scipy.io.savemat(savefolder + 'sequence_%06d.mat' % idx, D, do_compression=True)

    D = dict(imgRgb=sc.im.rgb)
    scipy.io.savemat(savefolder + 'rgb_%06d.mat' % idx, D, do_compression=True)

    D = dict(imgNormals=sc.im.normals)
    scipy.io.savemat(savefolder + 'surface_normals_%06d.mat' % idx, D, do_compression=True)

    D = dict(planeData=RR)
    scipy.io.savemat(savefolder + 'plane_data_%06d.mat' % idx, D, do_compression=True)


# now want to also save the 'planedata', or the data which is used in the object discovery...

# def process_sequence(sequence):

# save_location = savefolder + sequence['name'] + '.mat'

# I don't want whatever mask is currently being use