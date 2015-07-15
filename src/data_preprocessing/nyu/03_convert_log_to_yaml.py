# But first let's load the video and do a sanity render...
import math
import yaml
import numpy as np
import os
import cPickle as pickle
import sys
# here will load the log file and convert to my format...
import real_data_paths as paths

sys.path.append(os.path.expanduser('~/projects/shape_sharing/src/'))
from common import voxel_data


def dump_log(log, scene):
    with open(scene + '/' + 'poses.yaml', 'w') as f:
        yaml.dump(log, f)


def get_scene_pose(scene):
    with open(scene + '/scene_pose.yaml') as f:
        return yaml.load(f)


def combine_rot_and_trans(rot, trans):
    '''
    combines rotation and translation into one matrix
    '''
    scene_H = np.hstack((rot, trans[:, None]))
    scene_H = np.vstack((scene_H, np.array([0, 0, 0, 1])))
    return scene_H


def convert_log_to_canonical(frames, scene_pose):
    '''
    uses the pose of the scene to convert the poses of the frames to some
    kind of canonical pose...
    '''
    scene_R = np.array(scene_pose['R']).reshape(3, 3)
    scene_origin = np.dot(scene_R, -np.array(scene_pose['origin']))
    scene_H = combine_rot_and_trans(scene_R, scene_origin)

    for frame in frames:
        old_pose = np.array(frame['pose']).reshape(4, 4)
        new_pose = np.dot(scene_H, old_pose)
        frame['pose'] = new_pose.flatten().tolist()

    return frames


def init_log(scene_id):
    framedict = {}
    framedict['pose'] = np.eye(4).flatten().tolist()
    framedict['camera'] = 1
    framedict['frame'] = 0
    framedict['intrinsics'] = [582.6, 0.0, 313,
                            0.0, 582.6, 238,
                            0, 0, 1]
    framedict['id'] = '%02d_%04d' % (framedict['camera'], framedict['frame'])
    framedict['depth_scaling'] = float(2**16)
    framedict['image'] = '%s_raw.mat' % scene_id
    framedict['rgb'] = '%s.png' % scene_id
    framedict['mask'] = 'mask_%s.png' % scene_id
    return [framedict]


for sequence in paths.scenes:

    scene = sequence['folder'] + sequence['scene']

    scene_pose = get_scene_pose(scene)
    print "Doing ", scene

    log = init_log(sequence['scene'])
    log = convert_log_to_canonical(log, scene_pose)
    dump_log(log, scene)

    # creating a blank grid...
    grid = voxel_data.WorldVoxels()
    grid.set_origin(np.array([0, 0, 0]))
    grid.set_voxel_size(0.01)
    print "Size is ", scene_pose['size']
    shape = np.array(scene_pose['size']) / float(grid.vox_size)
    print "Shape is ", shape
    grid.V = np.zeros(shape)

    with open(scene + '/ground_truth_tsdf.pkl', 'w') as f:
        pickle.dump(grid, f, protocol=pickle.HIGHEST_PROTOCOL)
