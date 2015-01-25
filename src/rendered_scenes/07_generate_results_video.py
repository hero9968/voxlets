'''
What I want to do is to generate a nice sweep-around of the results in blender
'''
import subprocess as sp
import sys
import os
import numpy as np
import yaml
import shutil

# following for text on images
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw

sys.path.append(os.path.expanduser('~/projects/shape_sharing/src/'))
from common import paths
from common import voxel_data
from common import mesh

full = True

if sys.platform == 'darwin':
    blender_path = "/Applications/blender.app/Contents/MacOS/blender"
    font_path = "/Library/Fonts/Verdana.ttf"
elif sys.platform == 'linux2':
    blender_path = "blender"
    font_path = "/usr/share/fonts/truetype/msttcorefonts/verdana.ttf"
else:
    raise Exception("Unknown platform...")

temp_path = '/tmp/video_images/'

frame_num = 0


def add_frame(pil_image):
    global frame_num
    frame_num += 1
    print "Adding frame %d" % frame_num
    save_path = temp_path + 'img%03d.png' % frame_num
    print save_path
    img.save(save_path)


font = ImageFont.truetype(font_path, 40)
fontsmall = ImageFont.truetype(font_path, 20)


def text_on_image(img, heading="", subhead1="", subhead2=""):
    draw = ImageDraw.Draw(img)
    draw.text((0, 0), heading, font=font, fill=255)
    draw.text((0, 420), subhead1, font=fontsmall, fill=150)
    draw.text((0, 450), subhead2, font=fontsmall, fill=150)
    return img


with open(paths.yaml_test_location, 'r') as f:
    test_sequences = yaml.load(f)

# main outer loop - doing each sequence
for sequence in test_sequences:

    # creating an empty temporary folder
    if os.path.isdir(temp_path):
        shutil.rmtree(temp_path)
    os.makedirs(temp_path)

    # load voxels of the results
    test_seq_path = '../../data/rendered_arrangements/test_sequences/%s/' % \
        sequence['name']
    scene_path = '../../data/rendered_arrangements/renders/'

    git_label = sp.check_output(["git", "describe"]).strip()

    # getting the paths to the different things to load:
    with open(test_seq_path + 'info.yaml', 'r') as f:
        sequence_data = yaml.load(f)
    scene_name = sequence_data['scene']

    names = ['ground_truth', 'visible_voxels', 'prediction']
    files_to_load = [scene_path + scene_name + '/voxelgrid.pkl',
                     test_seq_path + 'input_fusion.pkl',
                     test_seq_path + 'prediction.pkl']
    levels = [0, 0, 0]

    # adding input images to file
    with open(scene_path + scene_name + '/poses.yaml', 'r') as f:
        poses = yaml.load(f)

    for idx, frame in enumerate(sequence_data['frames']):
        impath = scene_path + scene_name + '/images/colour_' + \
            poses[frame]['id'] + '.png'

        img = Image.open(impath)
        img = text_on_image(img, "Input view %d" % idx)

        add_frame(img)
        add_frame(img)

    for name, filename, level in zip(names, files_to_load, levels):

        prediction = voxel_data.load_voxels(filename)

        # convert to mesh and save to obj file
        if full:
            print "Converting to mesh"
            ms = mesh.Mesh()
            ms.from_volume(prediction, level)
            ms.write_to_obj('/tmp/temp.obj')
            #ms.write_to_obj('/tmp/temp%s.obj' % name)

            # run blender, while giving the path to the mesh to load
            print "Rendering"
            sp.call([blender_path, "spinaround/spin.blend",
                     "-b", "-P", "spinaround/blender_spinaround.py"])

        print "Adding text to frames"

        for idx in range(1, 21):

            imgfile = "/tmp/%04d.png" % idx

            img = Image.open(imgfile).convert('L')
            img = text_on_image(
                img, name, "Sequence: %s" % sequence['name'], git_label)

            print "Writing frame %s %s %f" % (name, filename, level)
            add_frame(img)

            # repeat first frame a few times
            if idx == 1 or idx == 20:
                add_frame(img)
                add_frame(img)

    break
