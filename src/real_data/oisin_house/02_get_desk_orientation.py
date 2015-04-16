'''
strips points below the plane, and also above 1m or so above the plane
'''

from xml.dom import minidom
import os, sys
import numpy as np
sys.path.append(os.path.expanduser('~/projects/shape_sharing/src/'))
from common import mesh
import yaml
import real_data_paths as paths

inlier_threshold = 5  # what considered inlier for plane?
box_height = 500  # in mm


def load_xyz_points(xml_filepath):
    '''
    loads xyz points from an xml file, as saved by meshlab pointpicker
    '''
    xmldoc = minidom.parse(xml_filepath)
    itemlist = xmldoc.getElementsByTagName('point')

    plane_xyz = [np.array([float(item.attributes['x'].value),
                           float(item.attributes['y'].value),
                           float(item.attributes['z'].value)])
                 for item in itemlist]

    return np.vstack(plane_xyz)


def fitplane(XYZ):
    XYZ = XYZ.T
    [rows,npts] = XYZ.shape

    if not rows == 3:
        print XYZ.shape
        raise Exception('data is not 3D')
        return None

    if npts <3:
        raise Exception('too few points to fit plane')
        return None

    # Set up constraint equations of the form  AB = 0,
    # where B is a column vector of the plane coefficients
    # in the form   b(1)*X + b(2)*Y +b(3)*Z + b(4) = 0.
    t = XYZ.T
    p = (np.ones((npts,1)))
    A = np.hstack([t,p])

    [u, d, v] = np.linalg.svd(A)        # Singular value decomposition.
    B = v[3,:];                         # Solution is last column of v.
    nn = np.linalg.norm(B[0:3])
    B = B / nn
    return B


def correct_plane_orientation(plane, xyz1):
    # now decide which direction is up on this plane, and flip accordingly...
    dotted = np.dot(plane, xyz1.T)
    # deciding correct orientation simply by looking at how many points are
    # on each side of the plane...
    correct_orientation = (dotted > 0).sum() > (dotted < 0).sum()

    if not correct_orientation:
        print "Correcting plane orientation..."
        plane *= -1
    else:
        print "Orientation seems ok"

    return plane


def project_point_to_plane(points1, plane):
    all_points = []
    for point in points1:
        t = np.dot(plane, point)
        all_points.append(point[:3] - t * plane[:3])
    return np.vstack(all_points)


def to_homogeneous(xyz):
    return np.hstack([xyz, np.ones((xyz.shape[0],1))])


def norm_v(vec):
    return vec / np.linalg.norm(vec)


for sequence in paths.scenes:

    scene = sequence['folder'] + sequence['scene']

    with open(scene + '/dump.voxels') as f:
        f.readline()
        voxel_size = float(f.readline().strip().split(' ')[1])
    print "Each voxel in the grid is %f" % voxel_size

    plane_path = scene + '/plane.pp'
    plane_xyz = load_xyz_points(plane_path)
    print "PLANE: Loaded points of shape " , plane_xyz.shape

    # fit a plane to these points...
    plane = fitplane(plane_xyz)
    print "Fitted plane: ", plane

    # # Load all the points...
    # ms = mesh.Mesh()
    # ms.load_from_obj(scene + '/dump.obj')
    # xyz1 = to_homogeneous(ms.vertices)

    # extents projected onto plane
    extents1 = to_homogeneous(plane_xyz)
    # points_on_plane = project_point_to_plane(extents1, plane)

    x = norm_v(plane_xyz[0] - plane_xyz[1])
    y_temp = norm_v(plane_xyz[2] - plane_xyz[1])
    z = norm_v(np.cross(x, y_temp))
    y = norm_v(np.cross(z, x))
    R = np.vstack((x, y, z))
    print "Found coordinate system ", R

    # getting the origin
    origin = plane_xyz[1] * voxel_size * 1000

    # push the origin 10 mm below the plane for breathing room
    origin -= z * 10

    # getting the size of the box...
    size_x = np.linalg.norm(plane_xyz[0] - plane_xyz[1]) * voxel_size * 1000
    size_y = np.linalg.norm(plane_xyz[2] - plane_xyz[1]) * voxel_size * 1000
    size_z = box_height

    size = np.array([size_x, size_y, size_z])

    # hacks... dont know why I need to do this but I do, apparently...
    plane[1] *= -1
    R[:, 1] *= -1
    origin[1] *= -1

    #converting mm to m
    origin /= 1000
    size /= 1000
    plane[-1] /= 1000

    print "origin", origin
    print np.linalg.det(R)

    scene_pose = dict(
        R=R.flatten().tolist(),
        origin=origin.tolist(),
        size=size.tolist(),
        voxel_size=voxel_size,
        plane=plane.tolist())

    # now writing these to a yaml file...
    with open(scene + '/scene_pose.yaml', 'w') as f:
        yaml.dump(scene_pose, f)

