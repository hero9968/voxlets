'''
What I want to do is to generate a nice sweep-around of the results in blender
'''
import subprocess as sp
import sys
import os
import numpy as np
import yaml
import shutil
import copy

# following for text on images
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw

sys.path.append(os.path.expanduser('~/projects/shape_sharing/src/'))
from common import paths
from common import voxel_data
from common import mesh

full = True  # do we do all the voxel grid saving etc?

temp_path = '/tmp/video_images/'

frame_num = 0  # counter for which frame we are on

git_label = sp.check_output(["git", "describe"]).strip()

test_result_type = 'oracle'

font = ImageFont.truetype(paths.font_path, 40)
fontsmall = ImageFont.truetype(paths.font_path, 20)


def text_on_image(img, heading="", subhead1="", subhead2=""):
    draw = ImageDraw.Draw(img)
    draw.text((0, 0), heading, font=font, fill=255)
    draw.text((0, 420), subhead1, font=fontsmall, fill=150)
    draw.text((0, 450), subhead2, font=fontsmall, fill=150)
    return img


def sequence_path(test_type, seq_name):
    return paths.Bricks.prediction % (test_type, seq_name)

# main outer loop - doing each sequence
for sequence in paths.RenderedData.get_scene_list():

    # creating an empty temporary folder
    if os.path.isdir(temp_path):
        shutil.rmtree(temp_path)
    os.makedirs(temp_path)

    # getting the paths to the different things to load:
    names = ['ground_truth', 'oracle_prediction']
    files_to_load = [paths.RenderedData.ground_truth_voxels(sequence),
                     sequence_path(test_result_type, sequence)]
    levels = [0.0, 0.0]

    for name, filename, level in zip(names, files_to_load, levels):

        prediction = voxel_data.load_voxels(filename)

        # convert to mesh and save to obj file
        if full:
            print "Converting to mesh"
            ms = mesh.Mesh()
            ms.from_volume(prediction, level)
            ms.write_to_obj('/tmp/temp.obj')
            # ms.write_to_obj('/tmp/temp%s.obj' % name)

            # run blender, while giving the path to the mesh to load
            print "Rendering"
            sp.call([paths.blender_path,
                     "../rendered_scenes/spinaround/spin.blend",
                     "-b", "-P",
                     "../rendered_scenes/spinaround/blender_spinaround_frame.py"])

        print "Adding text to frames"

        idx = 1

        imgfile = "/tmp/%04d.png" % idx

        img = Image.open('/tmp/.png').convert('L')
        img = text_on_image(
            img, name, "Sequence: %s" % sequence, git_label)
        save_path = paths.Bricks.prediction_frame % \
            (test_result_type, sequence, name)
        img.save(save_path)


#    break
