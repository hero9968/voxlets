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
import scipy.io

sys.path.append(os.path.expanduser('~/projects/shape_sharing/src/'))
sys.path.append(os.path.expanduser('~/projects/shape_sharing/src/intrinsic/'))
from common import images
from common import voxel_data
from common import carving
from common import paths

# doing a loop here to loop over all possible files...
for scenename in paths.rendered_primitive_scenes:

    input_data_path = paths.scenes_location + scenename + '/'

    vid = images.RGBDVideo()
    vid.load_from_yaml(input_data_path, 'poses.yaml')
    #vid.play()

    # initialise voxel grid (could add a helper function to make this more explicit...?)
    vox = voxel_data.WorldVoxels()
    vox.V = np.zeros((150, 150, 75), np.float32)
    vox.set_voxel_size(0.01)
    vox.set_origin((-0.75, -0.75, -0.03))

    print "Performing voxel carving"
    carver = carving.Fusion()
    carver.set_video(vid)
    carver.set_voxel_grid(vox)
    vox = carver.fuse()

    savepath = paths.scenes_location + scenename + 'voxelgrid.pkl'
    print "Saving using custom routine to location %s" % savepath
    vox.save(savepath)

    #savepath = paths.scenes_location + scenename + '/voxelgrid2.mat'
    #print "Also saving to %s" % savepath
    #scipy.io.savemat(savepath, dict(V=vox.V))
