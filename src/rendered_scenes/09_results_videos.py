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

full = False  # do we do all the voxel grid saving etc?

temp_path = '/tmp/video_images/'

frame_num = 0  # counter for which frame we are on

git_label = sp.check_output(["git", "describe"]).strip()

test_result_type = 'oma'


def add_frame(pil_image):
    global frame_num
    frame_num += 1
    print "Adding frame %d" % frame_num
    save_path = temp_path + 'img%03d.png' % frame_num
    print save_path
    pil_image.save(save_path)


font = ImageFont.truetype(paths.font_path, 40)
fontsmall = ImageFont.truetype(paths.font_path, 20)


def text_on_image(img, heading="", subhead1="", subhead2=""):
    draw = ImageDraw.Draw(img)
    draw.text((0, 0), heading, font=font, fill=255)
    draw.text((0, 420), subhead1, font=fontsmall, fill=150)
    draw.text((0, 450), subhead2, font=fontsmall, fill=150)
    return img


with open(paths.RenderedData.yaml_test_location, 'r') as f:
    test_sequences = yaml.load(f)


def sequence_path(test_type, seq_name):
    return paths.RenderedData.voxlet_prediction_path % (test_type, seq_name)

# main outer loop - doing each sequence
for sequence in paths.RenderedData.test_sequence():

    # creating an empty temporary folder
    if os.path.isdir(temp_path):
        shutil.rmtree(temp_path)
    os.makedirs(temp_path)

    # load voxels of the results

    # getting the paths to the different things to load:
    names = ['ground_truth', 'visible_voxels', 'prediction']
    files_to_load = [paths.RenderedData.ground_truth_voxels(sequence['scene']),
                     sequence_path('visible_voxels', sequence['name']),
                     sequence_path(test_result_type, sequence['name'])]
    levels = [0.0, 0.0, 0.0]

    # adding input images to file
    with open(paths.RenderedData.video_yaml(sequence['scene']), 'r') as f:
        poses = yaml.load(f)

    for idx, frame in enumerate(sequence['frames']):

        impath = paths.RenderedData.colour_path(
            sequence['scene'], poses[frame]['id'])

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
            # ms.write_to_obj('/tmp/temp%s.obj' % name)

            # run blender, while giving the path to the mesh to load
            print "Rendering"
            sp.call([paths.blender_path, "spinaround/spin.blend",
                     "-b", "-P", "spinaround/blender_spinaround.py"])

        print "Adding text to frames"

        for idx in range(1, 21):

            imgfile = "/tmp/%04d.png" % idx

            img = Image.open(imgfile).convert('L')
            img = text_on_image(
                img, name, "Sequence: %s" % sequence['name'], git_label)

            # print "Writing frame %s %s %f" % (name, filename, level)
            print "Writing frame ", idx
            add_frame(img)

            # repeat the first frame a few times at the end
            if idx == 1:
                print "Adding forst frame"
                first_frame = img
                add_frame(img)
                add_frame(img)

            if idx == 20:
                add_frame(first_frame)
                add_frame(first_frame)

    moviesavefile = paths.RenderedData.voxlets_path + '/predictions/' + test_result_type + \
        '/' + sequence['name'] + '.mp4'
    # here need to run the video combining algorithm
    sp.call(("ffmpeg",
             "-f", "image2",
             "-r", "5",
             "-i", temp_path + "img%03d.png",
             "-b:v", "1024k",
             "-vcodec", "mpeg4",
             "-y", moviesavefile))

    break