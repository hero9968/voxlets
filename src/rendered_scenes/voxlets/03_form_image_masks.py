'''
in the long run, move this into the rendering or something...
'''
import sys
import os
import numpy as np
import scipy

sys.path.append(os.path.expanduser('~/projects/shape_sharing/src/'))
sys.path.append(os.path.expanduser('~/projects/shape_sharing/src/intrinsic/'))

from common import images
from common import paths
from common import parameters

# setting the bounds of the cube...
grid_min, grid_max = parameters.RenderedVoxelGrid.aabb()
print grid_min, grid_max

# doing a loop here to loop over all possible files...
for scenename in paths.RenderedData.get_scene_list():

    print "Making masks for %s" % scenename

    vid = images.RGBDVideo()
    vid.load_from_yaml(paths.RenderedData.video_yaml(scenename))

    # now for each frame in the video, project the points into 3D and work out
    # which are within the specified cube...
    for frame in vid.frames:

        # remove points outside the bounding cuboid
        xyz = frame.get_world_xyz()
        inside = np.all(np.logical_and(xyz > grid_min, xyz < grid_max), axis=1)
        inside = inside.reshape(frame.depth.shape)

        # also remove points on the floor plane
        floor_height = parameters.RenderedVoxelGrid.origin[2]
        above_floor = np.abs(xyz[:, 2]) > floor_height
        above_floor = above_floor.reshape(frame.depth.shape)

        mask = np.logical_and(inside, above_floor)

        # choose the save location carefully
        savepath = paths.RenderedData.mask_path(scenename, frame.frame_id)
        scipy.misc.imsave(savepath, mask)
