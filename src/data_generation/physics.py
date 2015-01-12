import bpy
import sys, os
import random
import string
import shutil
import numpy as np
obj_folderpath = os.path.expanduser('~/projects/shape_sharing/data/meshes/primitives/ply_files/')
save_path = os.path.expanduser('~/projects/shape_sharing/data/rendered_arrangements/renders/')

min_to_load = 3 # in future this will be random number
max_to_load = 10

camera_names = ['Camera', 'Camera.001', 'Camera.002']
frames_per_camera = [20, 20, 10]

with open(obj_folderpath + '../all_names.txt', 'r') as f:
    models_to_use = [line.strip() for line in f]

#######################################################

def write_pose(f, camera, frame, pose):
    '''
    writes pose to file f in yaml format
    '''
    f.write('-  image:  images/%02d_%04d.png\n' % (camera, frame))
    f.write('   id:     %02d_%04d\n' % (camera, frame))
    f.write('   camera: %02d\n' % camera)
    f.write('   frame:  %04d\n' % frame)
    f.write('   depth_scaling:  %f\n' % 4)

    print(pose)
    pose_string = '   pose:   ['
    for ii in range(4):
        for jj in range(4):
            pose_string += str(pose[ii][jj]) + ', '
    f.write(pose_string[:-2] + ']\n')

    # TODO: should instead do: http://blender.stackexchange.com/questions/16472/how-can-i-get-the-cameras-projection-matrix
    focal_length = 579.679 # = 320 / tand(57.8deg / 2)
    dx = 320 # = w/2
    dy = 240 # = h/2
    # note that matricies go across first!
    intrinsics_string = '   intrinsics: [%f, 0, %f,   0, %f, %f,   0, 0, 1]\n\n' % ( focal_length, dx, focal_length, dy)
    f.write(intrinsics_string)


def loadSingleObject(number):
    '''
    loads in a single object and puts it above the plane in a random place
    '''
    # setting tags of existing objs so we know which object we have loaded in
    for obj in bpy.data.objects:
        obj.tag = True
    
    # loading in the new model
    modelname = random.choice(models_to_use)
    filepath = obj_folderpath + modelname
    print(filepath)
    bpy.ops.import_mesh.ply(filepath=filepath)

    # finding which one we just loaded
    imported_object = [obj for obj in bpy.data.objects if obj.tag is False]

    # now moving it to a random position
    for obj in imported_object:
        print("Setting location")
        x = random.random() * 1.2 - 0.6
        y = random.random() * 1.2 - 0.6
        obj.location = (x, y, 5) # this will be a random position above the plane
        bpy.context.scene.objects.active = obj

    bpy.ops.rigidbody.object_add(type='ACTIVE')
    return imported_object[0]

    
def norm(X):
    return X / np.sqrt(np.sum(X**2))

def normalise_matrix(M):
    for i in range(3):
        M[i, :] = norm(M[i, :])
    return M

def renderScene(name):
    '''
    renders scene from all the rotations of all the cameras 
    also saves the camera pose matrices
    '''
    scene = bpy.data.scenes['Scene']

    pose_filename = save_path + name + '/poses.yaml'
    with open(pose_filename, 'w') as pose_file_handle:

        # looping over each elevation setting
        for count, (camera, frames) in enumerate(zip(camera_names, frames_per_camera)):

            scene.camera = bpy.data.objects[camera]
            scene.frame_end = frames

            # setting the final output filename and rendering
            #scene.render.filepath = save_path + name + '/' + str(count) + '_####.png'
            #CompositorNodeOutputFile.base_path = \
            scene.node_tree.nodes['File Output'].base_path = \
                save_path + name + '/' + str(count)

            # trying to fix the gamma bug here....
            scene.sequencer_colorspace_settings.name = 'Raw'
            scene.view_settings.view_transform = 'Raw'

            bpy.ops.render.render( write_still=True, animation=True )

            # saving the camera poses
            for frame in range(scene.frame_start, scene.frame_end+1):
                scene.frame_set(frame)
                pose_mat = np.array(scene.camera.matrix_world)

                # normalise matrix and convert to computer vision coordinate convention
                pose_mat[0:3, 0:3] = normalise_matrix(pose_mat[0:3, 0:3])
                pose_mat[0:3, 1] *= -1
                pose_mat[0:3, 2] *= -1

                print(np.linalg.det(pose_mat))
                write_pose(pose_file_handle, count + 1, frame, pose_mat)


#######################################################
 
# this is the overall filename, a random string of characters
filename = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(10))
if not os.path.exists(save_path + filename):
    os.makedirs(save_path + filename)
    os.makedirs(save_path + filename + '/images/')

loaded_objs = []

# choosing number to load
num_to_load = random.randint(min_to_load, max_to_load)

for ii in range(num_to_load):

    print("Loading object " + str(ii))

    # clearing the cache?\
    context = bpy.context # or whatever context you have
    bpy.ops.ptcache.free_bake_all({'scene': bpy.data.scenes['Scene']})
    scene = bpy.context.screen.scene
    context.scene.frame_set(scene.frame_start)    

    obj = loadSingleObject(ii)

    # following values chosen by trial and error...
    obj.rigid_body.linear_damping = 0.6
    obj.rigid_body.angular_damping = 0.75
    obj.rigid_body.friction = 0.1
    obj.rigid_body.collision_shape='CONVEX_HULL'

    scale = 0.25
    obj.scale = (scale, scale, scale)

    loaded_objs.append(obj)

# baking the cache
bpy.ops.ptcache.bake_all(bake=True)

# applying the transforms
scene = bpy.context.screen.scene
context = bpy.context # or whatever context you have
context.scene.frame_set(scene.frame_end)    
print(scene.frame_end)

# selecting just the object
bpy.ops.object.select_all(action='DESELECT')
for obj in loaded_objs:
    obj.select = True

bpy.ops.object.visual_transform_apply()

# need to remove physics from the objects or they will keep moving during the camera motion
for obj in loaded_objs:
    bpy.context.scene.objects.active = obj
    bpy.ops.rigidbody.object_remove()

# clearing the cache? (don't think I need to do this)
#bpy.ops.ptcache.free_bake_all({'scene': bpy.data.scenes['Scene']})

# moving the fish tank and the funnel out the way
bpy.data.objects['Cube.002'].location = (0, 0, 100)
bpy.data.objects['Cube.001'].location = (0, 0, 100)

renderScene(filename)

bpy.ops.wm.save_as_mainfile(filepath=save_path + filename + "/scene.blend")

'''todo here: move the files around to more sensible structure.'''
for count, frames in enumerate(frames_per_camera):
    for frame in range(frames):

        # moving the image
        source_path = save_path + filename + '/%d/Image%04d.png' % (count, frame + 1)
        dest_path = save_path + filename + '/images/%02d_%04d.png' % (count + 1, frame + 1)

        print("source path is  " + source_path)
        print("dest path is  " + dest_path)

        shutil.move(source_path, dest_path)

    os.rmdir(save_path + filename + '/%d/' % count)

quit()
#erg0s0f!
