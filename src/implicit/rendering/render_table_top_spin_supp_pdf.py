#
# Run from command line
# blender -b spin_tt.blend -P render_video_table_top_spin.py
#

import bpy
import sys
import os
import numpy as np

# folder of objs
ip_dir = '/media/ssd/data/oisin_house/implicit/models/'
op_dir = '/media/ssd/data/oisin_house/implicit/renders/'

if not os.path.isdir(op_dir):
    os.mkdir(op_dir)

views = np.radians([270, 180, 90, 0])

#scenes = ['saved_00207_[536]', 'saved_00230_[45]', 'saved_00231_[55]', 'saved_00233_[134]']
file_types = ['gt.png.obj', 'Medioid.png.obj', 'pred_remove_excess.png.obj', 'visible.png.obj']

# each tuple: (rendername, modelname, filename)
render_types = [('gt', 'rays_cobweb', 'gt_render.png.obj'),
                ('visible', 'rays_cobweb', 'visible_render.png.obj'),
                ('rays', 'rays', 'prediction_render.png.obj'),
                ('zheng2', 'zheng2', 'prediction_render.png.obj')]

# colors
colors = np.asarray([[168, 211, 36], [80, 192, 233], [255, 198, 65], [255, 95, 95], [203, 151, 255]]) / 255.0
                    # green, #blue, orange, red, purple

# create material
mat = bpy.data.materials.new("PKHG")
mat.diffuse_color = colors[0, :]

fast_render = True
if fast_render:
    bpy.data.scenes["Scene"].cycles.samples = 100

for render_type_idx, (rendername, modelname, filename) in enumerate(render_types):

    prediction_dir = ip_dir + modelname + '/predictions/'
    scenes = os.listdir(prediction_dir)
    print (scenes)

    for scene_name in scenes:
        # load obj file
        full_path_to_file = prediction_dir + scene_name + '/' + filename

        print(full_path_to_file)
        bpy.ops.import_scene.obj(filepath=full_path_to_file, axis_forward='X', axis_up='Z')

        # set material color
        print(colors[render_type_idx, :])
        mat.diffuse_color = colors[render_type_idx, :]

        o = bpy.context.selected_objects[0]
        o.active_material = mat
        o.rotation_mode = 'XYZ'

        # remove holes from mesh
        # mess fix this
        scene = bpy.data.scenes['Scene']
        for obj in scene.objects:
            if obj.type == 'MESH':
                scene.objects.active = obj

                bpy.ops.object.mode_set(mode='EDIT')
                bpy.ops.mesh.fill_holes()
                bpy.ops.object.mode_set(mode='OBJECT')

        for ii, v in enumerate(views):
            o.rotation_euler[2] = v

            # render it
            bpy.context.scene.render.filepath = \
                op_dir + scene_name + '_' + rendername + '_view_' + str(ii).zfill(3) + '.png'
            bpy.ops.render.render(write_still=True)

        # delete file
        bpy.ops.object.delete()