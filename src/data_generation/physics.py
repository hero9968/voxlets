import bpy
import sys, os
import random
import string

obj_folderpath = os.path.expanduser('~/projects/shape_sharing/data/meshes/primitives/ply_files/')
save_path = './data/renders/'

# seeding random
random.seed(20)

min_to_load = 3 # in future this will be random number
max_to_load = 10

with open(obj_folderpath + '../all_names.txt', 'r') as f:
    models_to_use = [line.strip() for line in f]

#######################################################

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

    
def renderScene():
    '''
    renders scene from all the rotations of all the cameras 
    also saves the camera pose matrices
    '''

    scene = bpy.data.scenes['Scene']
    camera_names = ['Camera', 'Camera.001', 'Camera.002']
    frames_per_camera = [2, 2, 2]

    # this is the overall filename
    name = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(6))

    # looping over each elevation setting
    for count, (camera, frames) in enumerate(zip(camera_names, frames_per_camera)):

        scene.camera = bpy.data.objects[camera]
        scene.frame_end = frames

        # setting the final output filename and rendering
        #scene.render.filepath = save_path + name + '/' + str(count) + '_####.png'
        #CompositorNodeOutputFile.base_path = \
        scene.node_tree.nodes['File Output'].base_path = \
            save_path + name + '/' + str(count)
        bpy.ops.render.render( write_still=True, animation=True )

        # saving the camera poses
        matrix_out = save_path + name + '/' + str(count) + '.txt'

        with open(matrix_out, 'w') as f:
            for frame in range(scene.frame_start, scene.frame_end+1):
                scene.frame_set(frame)
                mat = scene.camera.matrix_local
                write_str = '%f %f %f %f  %f %f %f %f  %f %f %f %f\n' % (mat[0][0], mat[0][1], mat[0][2], mat[0][3], mat[1][0], mat[1][1], mat[1][2], mat[1][3], mat[2][0], mat[2][1], mat[2][2], mat[2][3])
                f.write(write_str)

    return name


#######################################################
 

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

    obj.rigid_body.linear_damping = 0.5
    obj.rigid_body.angular_damping = 0.2
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

# applying the visual transform and setting its position to 
bpy.ops.object.visual_transform_apply()

for obj in loaded_objs:
    bpy.context.scene.objects.active = obj
    bpy.ops.rigidbody.object_remove()
# clearing the cache?
#bpy.ops.ptcache.free_bake_all({'scene': bpy.data.scenes['Scene']})

bpy.data.objects['Cube.002'].location = (0, 0, 100)
bpy.data.objects['Cube.001'].location = (0, 0, 100)

filename = renderScene()

bpy.ops.wm.save_as_mainfile(filepath=save_path + filename + "/scene.blend")

'''todo here: move the files around to more sensible structure.'''

quit()
#bpy.ops.export_scene.obj(filepath="data/scene.obj")

