'''
script to 'render' the bb objects front and back
does this by loading in the voxelised points and projecting them into image space
Unsure if I should be using the full size image, but i will for now - seems to be ok
'''
import os
from subprocess import call
import numpy as np
import sys
sys.path.append('/Users/Michael/projects/shape_sharing/src/3D/structured_train/')
from thickness import mesh
import scipy.io

base_path = '/Users/Michael/projects/shape_sharing/data/'
models_path = base_path + 'bigbird/models.txt'
views_path = base_path + "bigbird/poses_to_use.txt"

savefolder = 'bigbird_renders/'

rendered_shape = (1024, 1280)

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
    for view_line in views_f:
        view = view_line.strip()

        # setting up output paths
        out_path = model_folder + view + '_renders.mat'

        # doing the rendering
        print "Rendering..."
        front_face, back_face = render_images(vox_vertices, modelname, view)

        # saving to file
        print "Saving file " + out_path
        d = dict(front=front_face, back=back_face)
        scipy.io.savemat(out_path, d)

        #break

    print "Done model " + modelname


