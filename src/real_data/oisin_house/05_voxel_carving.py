import numpy as np
import real_data_paths as paths
import real_params as parameters
import cPickle as pickle
import os, sys
import yaml
from time import time

sys.path.append(os.path.expanduser('~/projects/shape_sharing/src/'))
from common import images
from common import voxel_data
from common import carving

def get_scene_pose(scene):
    with open(scene + '/scene_pose.yaml') as f:
        return yaml.load(f)

# doing a loop here to loop over all possible files...
def process_sequence(sequence):

    scene = sequence['folder'] + sequence['scene']

    # # ignore if the output file exists...
    # if os.path.exists(scene + '/ground_truth_tsdf.pkl'):
    #     return

    print "Processing ", scene
    vid = images.RGBDVideo()
    vid.load_from_yaml(scene + '/poses.yaml')

    # load the scene parameters...
    scene_pose = get_scene_pose(scene)
    vgrid_size = np.array(scene_pose['size'])
    voxel_size = parameters.voxel_size
    vgrid_shape = vgrid_size / voxel_size

    # initialise voxel grid (could add helper function to make it explicit...?)
    vox = voxel_data.WorldVoxels()
    vox.V = np.zeros(vgrid_shape, np.uint8)
    vox.set_voxel_size(voxel_size)
    vox.set_origin(np.array([0, 0, 0]))

    print "Performing voxel carving", scene
    carver = carving.Fusion()
    carver.set_video(vid)
    carver.set_voxel_grid(vox)
    vox, visible = carver.fuse(mu=parameters.mu, filtering=False, measure_in_frustrum=True)
    in_frustrum = carver.in_frustrum

    print "Saving...", scene
    print vox.V.dtype
    print visible.V.dtype
    print in_frustrum.V.dtype

    with open(scene + '/ground_truth_tsdf.pkl', 'w') as f:
        pickle.dump(vox, f, protocol=pickle.HIGHEST_PROTOCOL)

    with open(scene + '/visible.pkl', 'w') as f:
        pickle.dump(visible, f, protocol=pickle.HIGHEST_PROTOCOL)

    with open(scene + '/in_frustrum.pkl', 'w') as f:
        pickle.dump(visible, f, protocol=pickle.HIGHEST_PROTOCOL)


# need to import these *after* the pool helper has been defined
if parameters.multicore:
    import multiprocessing
    import functools
    pool = multiprocessing.Pool(4)
    mapper = pool.map
else:
    mapper = map

tic = time()
mapper(process_sequence, paths.scenes, chunksize=1)
print "In total took %f s" % (time() - tic)