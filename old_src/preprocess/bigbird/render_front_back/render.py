'''
script to 'render' the bb objects front and back
does this by loading in the voxelised points and projecting them into image space
Unsure if I should be using the full size image, but i will for now - seems to be ok
'''
import os
from subprocess import call
import numpy as np
import sys
sys.path.append(os.path.expanduser('~/projects/shape_sharing/src/structured_train/'))
from thickness import mesh
from thickness import paths
from thickness import voxel_data
import scipy.io
import socket



base_path = paths.base_path
models_path = base_path + 'bigbird/bb_to_use.txt'
views_path = base_path + "bigbird/poses_to_use.txt"

savefolder = 'bigbird_renders/'

rendered_shape = (480, 640)


def render_images(vox_vertices, modelname, view):
    '''
    returns two images, of the vox vox_vertices reprojected into the camera
    '''

    # set up camera
    cam = mesh.Camera()
    cam.load_bigbird_matrices(modelname, view)
    cam.adjust_intrinsic_scale(0.5) # to create half-sized images

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


def frustum_voxels(vox_vertices, modelname, view):
    '''
    transforms the voxel grid into a frustum voxel arrangement from the camera...
    '''

    # set up camera
    cam = mesh.Camera()
    cam.load_bigbird_matrices(modelname, view)

    # project points - just to find the closest and furthest point from camera!
    xy = cam.project_points(vox_vertices)
    furthest_point = np.max(xy[:, 2])
    closest_point = np.min(xy[:, 2])

    # now create a frustum grid for the camera.
    grid = voxel_data.FrustumGrid()
    # grid.

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
    out_path = base_path + savefolder + modelname + '/' + view + '_renders.mat'
    if os.path.exists(out_path):
        print "Skipping " + out_path
        return []

    front, back = render_images(vox_vertices, modelname, view)

    print "Saving file " + out_path
    d = dict(front=front, back=back)
    scipy.io.savemat(out_path, d)



if __name__=='__main__':

    from multiprocessing import Pool
    import itertools
    pool = Pool(processes=5)

    # do for all models
    models_f = open(models_path, 'r')
    for model_line in models_f:
        modelname = model_line.strip()

        # creating the file if it doesn't exist
        model_folder = base_path + savefolder + modelname + '/'
        if not os.path.exists(model_folder):
            print "Creating folder for " + modelname
            os.makedirs(model_folder)

        # see if the files exist before loading in the voxels
        DIR = base_path + savefolder + modelname
        filecount = len([name for name in os.listdir(DIR) if os.path.isfile(os.path.join(DIR, name))])
        if filecount >= 75:
            print "Skipping " + modelname + " as appear to have at least 75 files"

        # load in the voxels
        print "Loading from " + base_path + '/bigbird_meshes/' + modelname + '/meshes/voxelised.txt'
        vox_vertices = np.loadtxt(base_path + '/bigbird_meshes/' + modelname + '/meshes/voxelised.txt')
        print "Loaded voxels from " + modelname + " of size " + str(vox_vertices.shape)

        try:
            # do for all views
            views_f = open(views_path, 'r')
            all_views = [view_line.strip() for view_line in views_f]
         #   done = [render_and_save((vox_vertices, modelname, view)) for view in all_views ]
            zipped_arguments = itertools.izip(itertools.repeat(vox_vertices), 
                                            itertools.repeat(modelname),
                                            all_views)
            pool.map(render_and_save, zipped_arguments)
        except Exception,e: 
            print str(e)


        print "Done model " + modelname


