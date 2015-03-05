import bpy

# create material
mat = bpy.data.materials.new("PKHG")
mat.diffuse_color = (0.056,0.527,1.0)


# load obj file
full_path_to_file = '/tmp/temp_voxlet.obj'
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
bpy.context.scene.render.filepath = '/tmp/temp_voxlet.png'
bpy.ops.render.render(write_still=True)

# delete file
bpy.ops.object.delete()
