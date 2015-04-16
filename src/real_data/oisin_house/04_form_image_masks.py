'''
in the long run, move this into the rendering or something...
'''
import sys
import os
import numpy as np
import scipy
import yaml

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.append(os.path.expanduser('~/projects/shape_sharing/src/'))
from common import images
import real_data_paths as paths

def get_scene_pose(scene):
    with open(scene + '/scene_pose.yaml') as f:
        return yaml.load(f)

# doing a loop here to loop over all possible files...
for sequence in paths.scenes:

    scene = sequence['folder'] + sequence['scene']

    print "Making masks for %s" % scene

    vid = images.RGBDVideo()
    vid.load_from_yaml(scene + '/poses.yaml')

    # loading the scene pise,,,
    scene_pose = get_scene_pose(scene)

    grid_min = np.array([0, 0, 0])
    grid_max = np.array(scene_pose['size'])
    print grid_min, grid_max

    # now for each frame in the video, project the points into 3D and work out
    # which are within the specified cube...
    for frame in vid.frames:

        # remove points outside the bounding cuboid
        xyz = frame.get_world_xyz()
        print np.nanmin(xyz, axis=0), np.nanmax(xyz, axis=0)
        print grid_min, grid_max
        inside = np.all(np.logical_and(xyz > grid_min, xyz < grid_max), axis=1)
        inside = inside.reshape(frame.depth.shape)

        # # also remove points on the floor plane
        # floor_height = 0.025
        # above_floor = np.abs(xyz[:, 2]) > floor_height
        # above_floor = above_floor.reshape(frame.depth.shape)

        # mask = np.logical_and(inside, above_floor)
        mask = inside

        # choose the save location carefully
        mask[np.isnan(mask)] = 0
        savepath = scene + '/frames/mask_%s.png' % frame.frame_id
        scipy.misc.imsave(savepath, mask)
