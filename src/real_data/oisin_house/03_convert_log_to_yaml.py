# But first let's load the video and do a sanity render...
import yaml
import numpy as np
import os
import sys

# here will load the log file and convert to my format...
import real_data_paths as paths

log_name = 'log.txt'

sys.path.append(os.path.expanduser('~/projects/shape_sharing/src/'))
from common import transformations


def read_log(scene_path):

    frames = []
    f = open(scene_path + '/' + log_name, 'r')

    for l in f:
        # modifying M2...
        rotter = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

        framedict = {}
        data = map(float, l.strip().split())
        framedict['frame'] = int(data[0])
        framedict['timestamp'] = data[1]

        # vv this is good
        t = data[2:5]
        R = np.array(transformations.quaternion_matrix(data[5:]))
        R[:3, 3] = R[:3, 3] = np.dot(np.linalg.inv(R[:3, :3]), -np.array(t))
        R = np.dot(rotter, R)
        R[:, 2] *= -1
        R[:3, :3] = np.linalg.inv(R[:3, :3])
        R[1, -1] *= -1
        # ^^ this is good

        # print "Warning - using M2 instead of R"
        framedict['pose'] = R.flatten().tolist()
        framedict['camera'] = 1
        framedict['intrinsics'] = [573.679000, 0, 320.000000,   0, 573.679000, 240.000000,   0, 0, 1]
        framedict['id'] = '%02d_%04d' % (framedict['camera'], framedict['frame'])
        framedict['depth_scaling'] = float(2**16) / 1000
        framedict['image'] = 'frames/%05d.pgm' % framedict['frame']
        framedict['rgb'] = 'frames/%05d.ppm' % framedict['frame']

        frames.append(framedict)

    return frames


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


for sequence in paths.scenes:

    scene_name = sequence['folder'] + sequence['scene']

    scene_pose = get_scene_pose(scene_name)

    print "Doing ", scene_name

    log = read_log(scene_name)
    log = convert_log_to_canonical(log, scene_pose)
    dump_log(log, scene_name)
