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
box_height = 0.5  # in m

to_flip = ['0556', '0561', '0756']
to_flip2 = ['0556', '0561']

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



def to_homogeneous(xyz):
    return np.hstack([xyz, np.ones((xyz.shape[0],1))])


def norm_v(vec):
    return vec / np.linalg.norm(vec)


for sequence in paths.scenes:

    scene = sequence['folder'] + sequence['scene']

    # with open(scene + '/dump.voxels') as f:
    #     f.readline()
    #     voxel_size = float(f.readline().strip().split(' ')[1])
    # print "Each voxel in the grid is %f" % voxel_size

    plane_path = scene + ('/%s_plane.pp') % sequence['scene']
    plane_xyz = load_xyz_points(plane_path)
    print "PLANE: Loaded points of shape " , plane_xyz.shape
    # print plane_xyz

    # fit a plane to these points...
    plane = fitplane(plane_xyz)
    # print "Fitted plane: ", plane

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

    print sequence['scene']
    if sequence['scene'] in to_flip:
        print "FLip"
        temp = x
        x = y
        y = temp
        z *= -1
    R = np.vstack((x, y, z))

    # getting the origin
    origin = plane_xyz[1]
    print origin

    # push the origin 30 mm below the plane for breathing room
    origin -= z * 0.030

    # getting the size of the box...
    if sequence['scene'] in to_flip2:
        size_x = np.linalg.norm(plane_xyz[2] - plane_xyz[1])
        size_y = np.linalg.norm(plane_xyz[0] - plane_xyz[1])
    else:
        size_x = np.linalg.norm(plane_xyz[0] - plane_xyz[1])
        size_y = np.linalg.norm(plane_xyz[2] - plane_xyz[1])
    size_z = box_height

    size = np.array([size_x, size_y, size_z])

    # new making bigger...
    origin -= x * 0.1
    origin -= y * 0.1
    size[0] += 0.0
    size[1] += 0.1

    # hacks... dont know why I need to do this but I do, apparently...
    # plane[1] *= -1
    # R[:, 1] *= -1
    # origin[1] *= -1

    #converting mm to m
    # origin /= 1000
    # size /= 1000
    # plane[-1] /= 1000

    if sequence['scene'] == '0127':
        size[2] += 0.05
        size[0] += 0.1
        size[1] += 0.2
        # origin -= x*0.1
        origin -= y*0.1

    if sequence['scene'] == '0333':
        origin -= y*0.2
        origin -= x*0.1
        size[0] += 0.1
        size[1] += 0.1

    if sequence['scene'] == '0029':
        origin -= y * 0.1
        size[1] += 0.2

    scene_pose = dict(
        R=R.flatten().tolist(),
        origin=origin.tolist(),
        size=size.tolist(),
        plane=plane.tolist())

    # now writing these to a yaml file...
    with open(scene + '/scene_pose.yaml', 'w') as f:
        yaml.dump(scene_pose, f)

