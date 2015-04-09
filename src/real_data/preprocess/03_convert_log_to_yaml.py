# But first let's load the video and do a sanity render...
import math
import yaml
import numpy as np
import os

# here will load the log file and convert to my format...
import real_data_paths as paths

log_name = 'log.txt'


def quaternion_matrix(quaternion):
    """Return homogeneous rotation matrix from quaternion.

    >>> M = quaternion_matrix([0.99810947, 0.06146124, 0, 0])
    >>> numpy.allclose(M, rotation_matrix(0.123, [1, 0, 0]))
    True
    >>> M = quaternion_matrix([1, 0, 0, 0])
    >>> numpy.allclose(M, numpy.identity(4))
    True
    >>> M = quaternion_matrix([0, 1, 0, 0])
    >>> numpy.allclose(M, numpy.diag([1, -1, -1, 1]))
    True
    Assumes [w, x, y, z] I think...
    """
    _EPS = np.finfo(float).eps * 4.0

    q = np.array(quaternion, dtype=np.float64, copy=True)
    n = np.dot(q, q)
    if n < _EPS:
        return np.identity(4)
    q *= math.sqrt(2.0 / n)
    q = np.outer(q, q)
    return np.array([
        [1.0-q[2, 2]-q[3, 3],     q[1, 2]-q[3, 0],     q[1, 3]+q[2, 0], 0.0],
        [    q[1, 2]+q[3, 0], 1.0-q[1, 1]-q[3, 3],     q[2, 3]-q[1, 0], 0.0],
        [    q[1, 3]-q[2, 0],     q[2, 3]+q[1, 0], 1.0-q[1, 1]-q[2, 2], 0.0],
        [                0.0,                 0.0,                 0.0, 1.0]])


def read_log(scene_path):

    frames = []
    f = open(scene_path + '/' + log_name, 'r')
    # f2 = open(scene_path + '/log_M.txt', 'r')
    # for l, l2 in zip(f, f2):
    for l in f:
        # original = map(float, l2.strip().split())
        # print original[0]
        # print original[1:4]
        # M2 = np.array(original[4:]).reshape(4, 4)
        # modifying M2...
        rotter = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        # M3 = np.dot(M2, rotter)

        framedict = {}
        data = map(float, l.strip().split())
        framedict['frame'] = int(data[0])
        framedict['timestamp'] = data[1]

        # vv this is good
        t = data[2:5]
        R = np.array(quaternion_matrix(data[5:]))
        R[:3, 3] = R[:3, 3] = np.dot(np.linalg.inv(R[:3, :3]), -np.array(t))
        R = np.dot(rotter, R)
        R[:, 2] *= -1
        R[:3, :3] = np.linalg.inv(R[:3, :3])
        R[1, -1] *= -1

        # ^^ this is good



        # print "Warning - using M2 instead of R"
        framedict['pose'] = R.flatten().tolist()
        framedict['camera'] = 1
        framedict['intrinsics'] = [579.679000, 0, 320.000000,   0, 579.679000, 240.000000,   0, 0, 1]
        framedict['id'] = '%02d_%04d' % (framedict['camera'], framedict['frame'])
        framedict['depth_scaling'] = float(2**16) / 1000
        framedict['image'] = 'images/%05d.pgm' % framedict['frame']
        framedict['rgb'] = 'images/%05d.ppm' % framedict['frame']

        np.set_printoptions(precision=3, formatter={'float':lambda x: '%0.9f' % x})
        # if int(original[0]) == 110:
        #     print np.linalg.det(M3)
        #     print np.linalg.det(R)
        #     print t
        #     print '\n'
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


for sequence in paths.scenes: #['/Users/Michael/projects/shape_sharing/data/desks/test_scans/saved_00151/']:

    scene = sequence['folder'] + sequence['scene']

    scene_pose = get_scene_pose(scene)
    print scene_pose
    print "Doing ", scene

    log = read_log(scene)
    log = convert_log_to_canonical(log, scene_pose)
    dump_log(log, scene)
