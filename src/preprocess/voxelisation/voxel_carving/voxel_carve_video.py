'''
script to voxel carve a scene given a video of depth frames
in the long run this video should be very short, just with filepaths and paramters
the meat should be in another place

in the short run this will probably not be the case

TODO:
- TSDF fusion instead of straight voxel carving?
- GPU for speed? (probably too much!)
'''
import sys, os
import numpy as np
import scipy

sys.path.append(os.path.expanduser('~/projects/shape_sharing/src/'))
from common import images
from common import voxel_data
from common import carving

input_data_path = os.path.expanduser('~/projects/shape_sharing/src/data_generation/data/renders/RCLGCYQALN/')
pose_filename = 'poses.yaml'

vid = images.RGBDVideo()
vid.load_from_yaml(input_data_path, pose_filename)
#vid.play()

# initialise voxel grid (could add a helper function to make this more explicit...?)
vox = voxel_data.WorldVoxels()
vox.V = np.zeros((150, 150, 75), np.float32)
vox.set_voxel_size(0.01)
vox.set_origin((-0.75, -0.75, 0))

carver = carving.Carver()
carver.set_video(vid)
carver.set_voxel_grid(vox)
vox = carver.carve()

# save here! (how? frame, grid...?)
savepath = './data/grid.mat'
print "Saving to %s" % savepath
D = dict(grid=vox.V)
scipy.io.savemat(savepath, D)

