'''
script to 'render' the bb objects front and back
does this by loading in the voxelised points and projecting them into image space
Unsure if I should be using the full size image, but i will for now - seems to be ok
'''
import os
from subprocess import call
import numpy as np
import sys
sys.path.append(os.path.expanduser('~/projects/shape_sharing/src/3D/structured_train/'))
from thickness import mesh
from thickness import paths
import scipy.io
import socket
from multiprocessing import Pool
import itertools



base_path = paths.base_path
models_path = base_path + 'bigbird/models.txt'
views_path = base_path + "bigbird/poses_to_use.txt"

savefolder = 'bigbird_renders/'

rendered_shape = (1024, 1280)

pool = Pool(processes=10)

def render_images(vox_vertices, modelname, view):
    '''
    returns two images, of the vox vox_vertices reprojected into the camera
    '''

    # set up camera
    cam = mesh.Camera()
    cam.load_bigbird_matrices(modelname, view)

    # reproject points
    xy = cam.project_points(vox_vertices)

    # render two images
    rows = np.round(xy[:, 1])
    cols = np.round(xy[:, 0])

    back_face = np.zeros(rendered_shape)
    front_face = np.zeros(rendered_shape) + 1000000

    for row, col, depth in zip(rows, cols, xy[:, 2]):
        back_face[row, col] = np.maximum(back_face[row, col],depth)
        front_face[row, col] = np.minimum(front_face[row, col],depth)

    front_face[front_face==1000000] = np.nan
    back_face[back_face==0] = np.nan

    return front_face, back_face


def render_and_save(all_feats):

    vox_vertices, modelname, view = all_feats

    # setting up output paths
    out_path = model_folder + view + '_renders.mat'
    if os.path.exists(out_path):
        print "Skipping " + out_path
        return []

    print "Rendering..."
    front, back = render_images(vox_vertices, modelname, view)

    print "Saving file " + out_path
    d = dict(front=front, back=back)
    scipy.io.savemat(out_path, d)



# do for all models
models_f = open(models_path, 'r')
for model_line in models_f:
    modelname = model_line.strip()

    # load in the voxels
    vox_vertices = np.loadtxt(base_path + '/bigbird_meshes/' + modelname + '/meshes/voxelised.txt')
    print "Loaded voxels from " + modelname + " of size " + str(vox_vertices.shape)

    model_folder = base_path + savefolder + modelname + '/'
    if not os.path.exists(model_folder):
        print "Creating folder for " + modelname
        os.makedirs(model_folder)

    # do for all views
    views_f = open(views_path, 'r')
    all_views = [view_line.strip() for view_line in views_f]
    #done = [render_and_save(vox_vertices, modelname, view) for view in all_views ]
    zipped_arguments = itertools.izip(itertools.repeat(vox_vertices), 
                                    itertools.repeat(modelname),
                                    all_views)
    pool.map(render_and_save, zipped_arguments)
    

    print "Done model " + modelname


