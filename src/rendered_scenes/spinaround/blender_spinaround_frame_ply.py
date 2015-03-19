'''
a file to run in blender to visualise results, by spinning the camera around a mesh
'''

import bpy
import sys, os
import numpy as np

#meshpath = os.getenv('meshpath')
meshpath='/tmp/temp.ply'

# load the mesh from the meshpath
bpy.ops.import_mesh.ply(filepath=meshpath)
# render the sequence of results

scene = bpy.data.scenes['Scene']
scene.frame_end = 20

mat = bpy.data.materials.new('VertexMat')

mat.use_vertex_color_light= True

scene.objects['temp'].data.materials.append(mat)

# setting the final output filename and rendering
#scene.render.filepath = save_path + name + '/' + str(count) + '_####.png'
#CompositorNodeOutputFile.base_path = \
#scene.node_tree.nodes['File Output'].base_path = '/'
#    save_path + name + '/' + str(count)

#scene.node_tree.nodes['File Output.001'].base_path = \
#    save_path + name + '/' + str(count)

bpy.ops.render.render(write_still=True, animation=False)
