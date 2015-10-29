'''
a file to run in blender to visualise results, by spinning the camera around a mesh
'''

import bpy
import sys, os
import numpy as np

quick_render = False

#meshpath = os.getenv('meshpath')
meshpath= os.getenv('BLENDERSAVEFILE') + '.obj'
savepath = os.getenv('BLENDERSAVEFILE')

# load the mesh from the meshpath
bpy.ops.import_scene.obj(filepath=meshpath, axis_forward='X', axis_up='Z')
# render the sequence of results

scene = bpy.data.scenes['Scene']
scene.frame_end = 20

# remove holes from mesh
for obj in scene.objects:
    if obj.type == 'MESH':
        scene.objects.active = obj

        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.fill_holes()
        bpy.ops.object.mode_set(mode='OBJECT')


# setting the final output filename and rendering
scene.render.filepath = savepath
#CompositorNodeOutputFile.base_path = \
#scene.node_tree.nodes['File Output'].base_path = '/'
#    save_path + name + '/' + str(count)

#scene.node_tree.nodes['File Output.001'].base_path = \
#    save_path + name + '/' + str(count)

if quick_render:
    # makes for a grainy render, but much quicker!
    scene.cycles.samples = 40

bpy.ops.render.render(write_still=True, animation=False)
