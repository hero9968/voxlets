import bpy
import sys, os
import random

obj_folderpath = os.path.expanduser('~/projects/shape_sharing/data/meshes/models/')
models_to_use = ['1049af17ad48aaeb6d41c42f7ade8c8.obj',
'109d55a137c042f5760315ac3bf2c13e.obj']

#bpy.data.scenes['Scene'].frame_end = 500

#######################################################

def loadSingleObject():
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
    bpy.ops.import_scene.obj(filepath=filepath)
    
    # finding which one we just loaded
    imported_object = [obj for obj in bpy.data.objects if obj.tag is False]
    
    # now moving it to a random position
    for obj in imported_object:
        print("Setting location")
        obj.location = (0, 0, 10) # this will be a random position above the plane
        bpy.context.scene.objects.active = obj
        
    bpy.ops.rigidbody.object_add(type='ACTIVE')
    return imported_object[0]

    
def renderScene():
    bpy.data.scenes['Scene'].render.filepath = 'data/render.png'
    bpy.ops.render.render( write_still=True )


#######################################################
 
num_to_load = 8 # in future this will be random number

for ii in range(num_to_load):

    print("Loading object " + str(ii))

    # clearing the cache?
    bpy.ops.ptcache.free_bake_all({'scene': bpy.data.scenes['Scene']})

    obj = loadSingleObject()
    
    # baking the cache
    bpy.ops.ptcache.bake_all(bake=True)

    # applying the transforms
    scene = bpy.context.screen.scene
    context = bpy.context # or whatever context you have
    context.scene.frame_set(scene.frame_end)    
    print(scene.frame_end)

    # selecting just the object
    bpy.ops.object.select_all(action='DESELECT')
    obj.select = True

    # applying the visual transform and setting its position to 
    bpy.ops.object.visual_transform_apply()
    obj.rigid_body.type = 'PASSIVE'

    # clearing the cache?
    bpy.ops.ptcache.free_bake_all({'scene': bpy.data.scenes['Scene']})

    
    

renderScene()

#bpy.ops.export_scene.obj(filepath="data/scene.obj")

