'''
in the long run, move this into the rendering or something...
'''
import sys
import os
import numpy as np
import scipy
import yaml
from xml.dom import minidom


import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.append(os.path.expanduser('~/projects/shape_sharing/src/'))
from common import images, mesh
import real_data_paths as paths

def get_scene_pose(scene):
    with open(scene + '/scene_pose.yaml') as f:
        return yaml.load(f)


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


# doing a loop here to loop over all possible files...
for sequence in paths.scenes:

    scene = sequence['folder'] + sequence['scene']

    print "Making masks for %s" % scene

    vid = images.RGBDVideo()
    vid.load_from_yaml(scene + '/poses.yaml')

    xyz = vid.frames[0].reproject_3d().T
    xyz = xyz[~np.any(np.isnan(xyz), axis=1), :]
    print xyz.shape

    xyz_points = load_xyz_points(scene + '/%s_plane.pp' % sequence['scene'])
    print xyz_points

    # here need to make a face for each of these...
    new_points = np.vstack((xyz_points,
               xyz_points + np.array([0, 0, 0.2]),
               xyz_points + np.array([0, 0.2, 0])
               ))

    new_faces = np.arange(12).reshape(3, 4).T


    savepath = scene + '/points.obj'
    ms = mesh.Mesh()
    ms.vertices = np.vstack((new_points, xyz))
    ms.faces = new_faces


    ms.write_to_obj(savepath)

