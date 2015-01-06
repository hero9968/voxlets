import bpy
import sys, os
import random
import string

obj_folderpath = os.path.expanduser('~/projects/shape_sharing/data/meshes2/models/')

# seeding random
random.seed(121)

num_to_load = 10 # in future this will be random number

with open(obj_folderpath + '../all_names.txt', 'r') as f:
    models_to_use = [f.readline().strip() for ii in range(200)]
print(models_to_use)

#models_to_use = ['1049af17ad48aaeb6d41c42f7ade8c8.obj']#,
#'109d55a137c042f5760315ac3bf2c13e.obj']

#bpy.data.scenes['Scene'].frame_end = 500

# z = 13
# x, y = -5 ... +5

#######################################################

def loadSingleObject(number):
    '''
    loads in a single object and puts it above the plane in a random place
    '''

    #should clear selection here

    # setting tags of existing objs so we know which object we have loaded in
    for obj in bpy.data.objects:
        obj.tag = True
    
    # loading in the new model
    modelname = random.choice(models_to_use)
    filepath = obj_folderpath + modelname
    bpy.ops.import_scene.obj(filepath=filepath, axis_forward='X', axis_up='Z')
    
    # finding which one we just loaded
    imported_object = [obj for obj in bpy.data.objects if obj.tag is False]
    
    # now moving it to a random position
    for obj in imported_object:
        print("Setting location")
        x = random.random() * 1.2 - 0.6
        y = random.random() * 1.2 - 0.6
        obj.location = (x, y, 14) # this will be a random position above the plane
        bpy.context.scene.objects.active = obj
    
    bpy.ops.rigidbody.object_add(type='ACTIVE')
    return imported_object[0]

    
def renderScene():
    name = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(6))
    bpy.data.scenes['Scene'].render.filepath = 'data/' + name + '.png'
    bpy.ops.render.render( write_still=True )
    return name


#######################################################
 

loaded_objs = []

for ii in range(num_to_load):

    print("Loading object " + str(ii))

    # clearing the cache?\
    context = bpy.context # or whatever context you have
    bpy.ops.ptcache.free_bake_all({'scene': bpy.data.scenes['Scene']})
    scene = bpy.context.screen.scene
    context.scene.frame_set(scene.frame_start)    

    obj = loadSingleObject(ii)

    obj.rigid_body.linear_damping = 0.5
    obj.rigid_body.angular_damping = 0.1
    scale = 4
    obj.scale = (scale, scale, scale)
    obj.rigid_body.collision_shape='CONVEX_HULL'

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

# clearing the cache?
bpy.ops.ptcache.free_bake_all({'scene': bpy.data.scenes['Scene']})

filename = renderScene()

bpy.ops.wm.save_as_mainfile(filepath="data/" + filename + ".blend")
quit()
#bpy.ops.export_scene.obj(filepath="data/scene.obj")

