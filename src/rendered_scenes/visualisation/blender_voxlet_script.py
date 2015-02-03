import bpy
import sys, os
#import random
#import string
#import shutil
#import numpy as np
sys.path.append(os.path.expanduser('~/projects/shape_sharing/src/'))
#from common import paths
mesh_dir = os.path.expanduser('~/projects/shape_sharing/data/rendered_arrangements/voxlets/dictionary/visualisation/kmeans/')
print(mesh_dir)
op_dir = os.path.expanduser('~/projects/shape_sharing/data/rendered_arrangements/voxlets/dictionary/visualisation/kmeans/')
print(op_dir)

# create material
mat = bpy.data.materials.new("PKHG")
mat.diffuse_color = (0.056,0.527,1.0)

file_list = [each for each in os.listdir(mesh_dir) if each.endswith('marching_cubes.obj')]

print(file_list)


try:
    for file_name in file_list:

        # load obj file
        full_path_to_file = mesh_dir + file_name
        print("Loading " + full_path_to_file)
        bpy.ops.import_scene.obj(filepath=full_path_to_file)

        o = bpy.context.selected_objects[0]
        o.active_material = mat

        # rotate - 90
        o.rotation_euler[0] = 0.0

        # sub divide the surface
        o.modifiers.new("subd", type='SUBSURF')
        o.modifiers['subd'].levels = 3

        # make sure its only in the first layer
        o.layers[0] = True
        o.layers[1] = False
        o.layers[2] = False

        # render it
        bpy.context.scene.render.filepath = op_dir + file_name[:-3] + 'png'
        bpy.ops.render.render(write_still=True)

        # delete file
        bpy.ops.object.delete()

except:
    quit()
quit()
    #erg0s0f!


