'''
script to voxel carve a scene given a video of depth frames

TODO:
- GPU for speed? (probably too much!)
'''
import sys
import os
import numpy as np
from time import time

sys.path.append(os.path.expanduser('~/projects/shape_sharing/src/'))
from common import images
from common import voxel_data
from common import carving

import paths
import parameters

# doing a loop here to loop over all possible files...
def carve_grid(scenename):

    vid = images.RGBDVideo()
    vid.load_from_yaml(paths.RenderedData.video_yaml(scenename))

    # initialise voxel grid (could add helper function to make it explicit...?)
    vox = voxel_data.WorldVoxels()
    vox.V = np.zeros(parameters.RenderedVoxelGrid.shape, np.float32)
    vox.set_voxel_size(parameters.RenderedVoxelGrid.voxel_size)
    vox.set_origin(parameters.RenderedVoxelGrid.origin)

    print "Performing voxel carving"
    carver = carving.Fusion()
    carver.set_video(vid)
    carver.set_voxel_grid(vox)
    vox, visible = carver.fuse(mu=parameters.RenderedVoxelGrid.mu)

    savepath = paths.RenderedData.ground_truth_voxels(scenename)
    print "Saving using custom routine to location %s" % savepath
    vox.save(savepath)

    savepath = paths.RenderedData.visible_voxels(scenename)
    print "Saving using custom routine to location %s" % savepath
    visible.save(savepath)



if parameters.multicore:
    # need to import these *after* pool_helper has been defined
    import multiprocessing
    pool = multiprocessing.Pool(parameters.cores)
    mapper = pool.map
else:
    mapper = map

if __name__ == "__main__":

    tic = time()
    mapper(carve_grid, paths.RenderedData.get_scene_list())
    print "In total took %f s" % (time() - tic)
