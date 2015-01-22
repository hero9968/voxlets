'''
What I want to do is to generate a nice sweep-around of the results in blender
'''
import subprocess as sp
import sys, os
import numpy as np
import yaml

# following for text on images
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw

sys.path.append(os.path.expanduser('~/projects/shape_sharing/src/'))
from common import paths
from common import voxel_data
from common import mesh

# load voxels of the results
test_seq_path = '../../data/rendered_arrangements/test_sequences/dm779sgmpnihle9x/'
scene_path = '../../data/rendered_arrangements/renders/'

git_label = sp.check_output(["git", "describe"]).strip()

# getting the paths to the different things to load:
with open(test_seq_path + 'info.yaml', 'r') as f:
    sequence_data = yaml.load(f)
scene_name = sequence_data['scene']

names = ['ground_truth', 'visible_voxels', 'prediction']
files_to_load = [scene_path + scene_name + '/voxelgrid.pkl',
                test_seq_path + 'visible_voxels.pkl',
                test_seq_path + 'prediction.pkl']
levels = [0, 0.5, 0]

font = ImageFont.truetype("/Library/Fonts/Verdana.ttf", 40)
fontsmall = ImageFont.truetype("/Library/Fonts/Verdana.ttf", 20)


sequence = 'de3dmo4kd0'

rate = 3.3
outf = "test.mp4"
print "opening video file"
cmdstring = ('ffmpeg',
             '-y',
             '-r', '%d' % rate,
             '-f','image2pipe',
             '-vcodec', 'mjpeg',
             '-i', 'pipe:',
             '-vcodec', 'mpeg4',
             outf
             )

p = sp.Popen(cmdstring, stdin=sp.PIPE, shell=False)

#for i in range(10):
    #im = Image.fromarray(np.uint8(np.random.randn(100,100)))
    #p.stdin.write(im.tostring('jpeg','L'))

# adding input images to file
with open(scene_path + scene_name + '/poses.yaml', 'r') as f:
    poses = yaml.load(f)


for idx, frame in enumerate(sequence_data['frames']):
    impath = scene_path + scene_name + '/images/' + poses[frame]['id'] + '.png'
    temp = (np.asarray(Image.open(impath)).astype(np.float32)/(65525))*255
    temp = temp.astype(np.int8)
    img = Image.fromarray(temp)
    draw = ImageDraw.Draw(img)
    draw.text((0, 0), "Input image %d" % idx, font=font, fill=0)
    print img
    p.stdin.write(img.convert('L').tostring('jpeg','L'))
    p.stdin.write(img.convert('L').tostring('jpeg','L'))


full = True


for name, filename, level in zip(names, files_to_load, levels):

    prediction = voxel_data.load_voxels(filename)

    # convert to mesh and save to obj file
    if full:
        print "Converting to mesh"
        ms = mesh.Mesh()
        ms.from_volume(prediction, level)
        ms.write_to_obj('/tmp/temp.obj')

        # run blender, while giving the path to the mesh to load
        print "Rendering"
        sp.call(["/Applications/blender.app/Contents/MacOS/blender",
            "spinaround/spin.blend", "-b", "-P", "spinaround/blender_spinaround.py"])


    print "Adding text to frames"

    for idx in range(1, 21):

        imgfile = "/tmp/%04d.png" % idx

        img = Image.open(imgfile).convert('L')
        draw = ImageDraw.Draw(img)

        draw.text((0, 0), name, font=font, fill=255)
        draw.text((0, 420), "Sequence: %s" % sequence, font=fontsmall, fill=150)
        draw.text((0, 450), git_label, font=fontsmall, fill=150)

        print "Writing frame"
        np_array = np.asarray(img)
        np_array = np_array.astype(np.int8)
        print np_array.shape

        print np_array.shape
        img2 = Image.fromarray(np_array)
        p.stdin.write(img.convert('L').tostring('jpeg','L'))

        # repeat first frame a few times
        if idx == 1 or idx == 21:
            p.stdin.write(img.convert('L').tostring('jpeg','L'))
            p.stdin.write(img.convert('L').tostring('jpeg','L'))
            p.stdin.write(img.convert('L').tostring('jpeg','L'))


p.stdin.close()
