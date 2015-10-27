"""
blender data/render_cad.blend --background -P render_helper.py

Renders an object-segmented image of all the objects in a blender scene,
for the NYU CAD scenes
WARNING: Only designed to do up to 255 objects. Should be very easy to extend
this to more by using colours, but I am just doing black and white
"""
import bpy
import numpy as np
import math
import os
import shutil


def norm(X):
    return X / np.sqrt(np.sum(X**2))


def normalise_matrix(M):
    for i in range(3):
        M[i, :] = norm(M[i, :])
    return M


def quaternion_from_matrix(matrix, isprecise=False):
    """Return quaternion from rotation matrix.
    from http://www.lfd.uci.edu/~gohlke/code/transformations.py.html
    """
    M = np.array(matrix, dtype=np.float64, copy=False)[:4, :4]
    if isprecise:
        q = np.empty((4, ))
        t = np.trace(M)
        if t > M[3, 3]:
            q[0] = t
            q[3] = M[1, 0] - M[0, 1]
            q[2] = M[0, 2] - M[2, 0]
            q[1] = M[2, 1] - M[1, 2]
        else:
            i, j, k = 1, 2, 3
            if M[1, 1] > M[0, 0]:
                i, j, k = 2, 3, 1
            if M[2, 2] > M[i, i]:
                i, j, k = 3, 1, 2
            t = M[i, i] - (M[j, j] + M[k, k]) + M[3, 3]
            q[i] = t
            q[j] = M[i, j] + M[j, i]
            q[k] = M[k, i] + M[i, k]
            q[3] = M[k, j] - M[j, k]
        q *= 0.5 / math.sqrt(t * M[3, 3])
    else:
        m00 = M[0, 0]
        m01 = M[0, 1]
        m02 = M[0, 2]
        m10 = M[1, 0]
        m11 = M[1, 1]
        m12 = M[1, 2]
        m20 = M[2, 0]
        m21 = M[2, 1]
        m22 = M[2, 2]
        # symmetric matrix K
        K = np.array([[m00-m11-m22, 0.0,         0.0,         0.0],
                         [m01+m10,     m11-m00-m22, 0.0,         0.0],
                         [m02+m20,     m12+m21,     m22-m00-m11, 0.0],
                         [m21-m12,     m02-m20,     m10-m01,     m00+m11+m22]])
        K /= 3.0
        # quaternion is eigenvector of K that corresponds to largest eigenvalue
        w, V = np.linalg.eigh(K)
        q = V[[3, 0, 1, 2], np.argmax(w)]
    if q[0] < 0.0:
        np.negative(q, q)
    return q


# setting paths
folders = [
    {"objdir": "/home/michael/projects/shape_sharing/data/nyu_renders_split/",
    "savedir": "/home/michael/projects/shape_sharing/data/cleaned_3D/segmented_renders_no_walls/"},
    # {"objdir": "/home/michael/projects/shape_sharing/data/cleaned_3D/binvox_with_walls/",
    # "savedir": "/home/michael/projects/shape_sharing/data/cleaned_3D/renders_with_walls/"}
    ]

for folder in folders:

    objdir = folder['objdir']
    savedir = folder['savedir']

    matpath = "/home/michael/projects/shape_sharing/data/cleaned_3D/camera_matrices/"

    fnames = os.listdir(objdir)

    # loop over each file...
    for fname in fnames:

        print(fname)
        # if not fname.endswith('.obj'):
        #     continue

        # load the camera matrices
        mats = {}
        for matname in ['K', 'R']:

            csv_fname = matpath + fname + '_%s.csv' % matname
            print(csv_fname)
            mats[matname] = np.loadtxt(csv_fname, delimiter=',')

        # delete all existing meshes
        bpy.ops.object.select_by_type(type = 'MESH')
        bpy.ops.object.delete(use_global=False)

        # import the new meshes from the directory of meshes
        obj_loaddir = objdir + '/' + fname + '/'

        for obj_count, obj_fname in enumerate(os.listdir(obj_loaddir)):
            this_loadpath = obj_loaddir + obj_fname
            obj = bpy.ops.import_scene.obj(
                filepath=this_loadpath, axis_forward='Y', axis_up='Z')

        # now colour each object differently
        this_col = 0
        for obj in bpy.data.objects:
            # generate random colour for the object
            if obj.name == 'Plane':
                obj.active_material.use_shadeless = True
                temp = float(254) / float(255)
                obj.active_material.diffuse_color = (temp, temp, temp)
                continue
            elif obj.name.startswith('Lamp') or \
                    obj.name.startswith('Cube') or \
                    obj.name.startswith('Bezier') or \
                    obj.name.startswith('Camera'):
                continue
            else:
                print("Hi ", obj.name)

            # give this object a unique coloured material
            mat = bpy.data.materials.new("PKHG")
            mat.use_shadeless= True
            temp = float(this_col) / 255
            print(temp)
            mat.diffuse_color = (temp, temp, temp)
            obj.active_material = mat

            this_col += 1


        # move camera to correct location
        bpy.data.objects['Camera'].location = (0, 0, 0)
        bpy.data.objects['Camera'].rotation_mode = 'QUATERNION'
        bpy.data.objects['Camera'].rotation_quaternion = quaternion_from_matrix(mats['R'].T)

        # render to /tmp/
        bpy.ops.render.render(write_still=True, animation=False )

        # make a directry to save files to
        this_savedir = savedir + fname + '/'
        if not os.path.exists(this_savedir):
            os.makedirs(this_savedir)

        # now move the files to the correct location
        rgb_savepath = this_savedir + 'rgb.png'
        shutil.move('/tmp/ColourImage0016.png', rgb_savepath)
        #
        # depth_savepath = this_savedir + 'depth.png'
        # shutil.move('/tmp/Image0016.png', depth_savepath)

        # # also copy the mesh there...
        # shutil.copy(obj_loadpath, this_savedir + fname)

        # # take the pose from the camera...
        # scene = bpy.data.scenes['Scene']
        # pose_mat = np.array(scene.camera.matrix_world)
        #
        # pose_mat[0:3, 0:3] = normalise_matrix(pose_mat[0:3, 0:3])
        # pose_mat[0:3, 1] *= -1
        # pose_mat[0:3, 2] *= -1
        #
        # # save the pose to a file somewhere...
        # with open(this_savedir + 'cam_pose.csv', 'w') as f:
        #     f.write(','.join(map(str, pose_mat.ravel().tolist())))

            # write_pose(pose_file_handle, count + 1, frame, pose_mat)
