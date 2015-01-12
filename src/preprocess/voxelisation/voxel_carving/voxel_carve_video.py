'''
script to voxel carve a scene given a video of depth frames
in the long run this video should be very short, just with filepaths and paramters
the meat should be in another place

in the short run this will probably not be the case
'''

import sys, os
import numpy as np
import sys
sys.path.append(os.path.expanduser('~/projects/shape_sharing/src/'))
import scipy
import matplotlib.pyplot as plt

from common import images
from common import voxel_data

input_data_path = '/Users/Michael/projects/shape_sharing/src/data_generation/data/renders/RCLGCYQALN/'
pose_filename = 'poses.yaml'

vid = images.RGBDVideo()
vid.load_from_yaml(input_data_path, pose_filename)
#vid.play()

# here want to do voxel carving

# initialise voxel grid (could add a helper function to make this more explicit...?)
vox = voxel_data.WorldVoxels()
vox.V = np.zeros((150, 150, 75), np.float32)
vox.set_voxel_size(0.01)
vox.set_origin((-0.75, -0.75, 0))


#plt.imshow(im.depth)
#plt.colorbar()
#plt.show()
'''Observation - depth from camera doesn't depend on application of intrinsics'''

# for each camera, project voxel grid into camera and see which ahead/behind of depth image
# (TSDF?)
for count, im in enumerate(vid.frames):

    print "\nFrame number %d with name %s" % (count, im.frame_id)
    print im.frame_id
 
    # Projecting voxels into image
    xyz = vox.world_meshgrid()
    projected_voxels = im.cam.project_points(xyz)

    print np.linalg.inv(im.cam.H)
    
    print np.min(projected_voxels[:, 2])
    print np.max(projected_voxels[:, 2])

    # now work out which voxels are in front of or behind the depth image
    # and location in camera image of each voxel
    uv = np.round(projected_voxels[:, :2]).astype(int)
    inside_image = np.logical_and.reduce((uv[:, 0] >= 0, uv[:, 1] >= 0, uv[:, 1] < im.depth.shape[0], uv[:, 0] < im.depth.shape[1]))
    all_observed_depths = im.depth[uv[inside_image, 1], uv[inside_image, 0]]

    print "There are %d voxels projected inside the image" % np.sum(inside_image)

    # doing the voxel carving
    known_to_be_empty = all_observed_depths > projected_voxels[inside_image, 2]
    known_to_be_empty_global_idx = np.where(inside_image)[0][known_to_be_empty]
    known_to_be_empty_global_sub = np.array(np.unravel_index(known_to_be_empty_global_idx, vox.V.shape))

    current_vals = vox.get_idxs(known_to_be_empty_global_sub.T)
    
    vox.set_idxs(known_to_be_empty_global_sub.T, current_vals + 1)

    print "Known empty is " + str(np.sum(known_to_be_empty)) + " out of " + str(known_to_be_empty.shape)


# save here! (how? frame, grid...?)
D = dict(grid=vox.V, proj=projected_voxels, inside_image=inside_image)
scipy.io.savemat('./data/grid.mat', D)


