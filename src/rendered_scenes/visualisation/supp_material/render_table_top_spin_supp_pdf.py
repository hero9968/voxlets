#
# Run from command line
# blender -b spin_tt.blend -P render_video_table_top_spin.py
#

import bpy
import sys
import os
import numpy as np

# folder of objs
ip_dir = '/home/omacaodh/prism/data5/projects/depth_completion/predictions/wednesday/oisin_house_full_training_data_mixed_voxlets_2/'
op_dir = '/home/omacaodh/Projects/depth_prediction/rendering/turn_table_iccv_video_pdf_01/'
if not os.path.isdir(op_dir):
    os.mkdir(op_dir)

#views = [0, 90, 180, 270]  # for spin around start at 270, end at -90
views = np.radians([270, 180, 90, 0])

# load files
scenes = os.listdir(ip_dir)
#scenes = ['saved_00207_[536]', 'saved_00230_[45]', 'saved_00231_[55]', 'saved_00233_[134]']
file_types = ['gt.png.obj', 'Medioid.png.obj', 'pred_remove_excess.png.obj', 'visible.png.obj']

# colors
colors = np.asarray([[168, 211, 36], [80, 192, 233], [255, 198, 65], [255, 95, 95], [203, 151, 255]]) / 255.0
                    # green, #blue, orange, red, purple

# create material
mat = bpy.data.materials.new("PKHG")
mat.diffuse_color = colors[0, :]

fast_render = True
if fast_render:
    bpy.data.scenes["Scene"].cycles.samples = 100

for scene_name in scenes:
    for file_name in file_types:
        # load obj file
        full_path_to_file = ip_dir + scene_name + '/' + file_name
        print(full_path_to_file)
        bpy.ops.import_scene.obj(filepath=full_path_to_file, axis_forward='X', axis_up='Z')

        o = bpy.context.selected_objects[0]
        o.active_material = mat
        o.rotation_mode = 'XYZ'

        # set material color
        if full_path_to_file.find('gt.png.obj') > 0:
            mat.diffuse_color = colors[0, :]
        elif full_path_to_file.find('pred_remove_excess.png.obj') > 0:  # voxlets
            mat.diffuse_color = colors[1, :]
        elif full_path_to_file.find('visible.png.obj') > 0:  #visible
            mat.diffuse_color = colors[2, :]
        elif full_path_to_file.find('Medioid.png.obj') > 0:
            mat.diffuse_color = colors[4, :]

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
            bpy.context.scene.render.filepath = op_dir + scene_name + '_' + file_name[:-8] + '_view_' + str(ii).zfill(3) + '.png'
            bpy.ops.render.render(write_still=True)

        # delete file
        bpy.ops.object.delete()
        # for ob in bpy.context.scene.objects:
        #     ob.select = ob.type == 'MESH' and ob.name.startswith(file_name[:-4])
        # bpy.ops.object.delete()

