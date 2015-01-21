'''
a file to run in blender to visualise results, by spinning the camera around a mesh
'''

import bpy
import sys, os
import numpy as np

#meshpath = os.getenv('meshpath')
meshpath='/tmp/temp.obj'

# load the mesh from the meshpath
bpy.ops.import_scene.obj(filepath=meshpath, axis_forward='X', axis_up='Z')
# render the sequence of results

scene = bpy.data.scenes['Scene']
scene.frame_end = 20

# setting the final output filename and rendering
#scene.render.filepath = save_path + name + '/' + str(count) + '_####.png'
#CompositorNodeOutputFile.base_path = \
#scene.node_tree.nodes['File Output'].base_path = '/'
#    save_path + name + '/' + str(count)

#scene.node_tree.nodes['File Output.001'].base_path = \
#    save_path + name + '/' + str(count)

bpy.ops.render.render(write_still=True, animation=True )
