import yaml
import os, shutil, glob
from scipy.misc import imread, imsave

synthetic = True

# make and empty folder

savefolder = 'synth_implicit_ims/'
source_dir = '/media/ssd/data/rendered_arrangements/implicit/renders/'
orig_dir = '/media/ssd/data/rendered_arrangements/renders/'
yaml_file = '/media/ssd/data/rendered_arrangements/splits/test.yaml'


files = glob.glob(savefolder + '*')
for f in files:
    os.remove(f)


A = '\includegraphics[width=\\synth_implicit_width, clip=true, trim=10 10 10 10]{'


test_data = yaml.load(open(yaml_file))

f = open('table.tex', 'w')

# print "Warning - only using a subset of the data"
for count, sequence in enumerate(test_data[:15]):
    if sequence['name'].startswith('az') or sequence['name'].startswith('rbr') or \
    sequence['name'].startswith('kt') or sequence['name'].startswith('4z') \
     or sequence['name'].startswith('eu') or sequence['name'].startswith('3d'):
        continue

    print sequence

    frames = yaml.load(open(orig_dir + sequence['scene'] + '/poses.yaml'))

    # input view image
    f.write(A + savefolder + sequence['name'] + '} & \n')

    # now copy the input view image to the output folder
    frame_num = int(sequence['frames'][0])
    print frames[frame_num]
    impath = orig_dir + sequence['scene'] + '/images/colour_' + frames[frame_num]['id'] + '.png'
    im = imread(impath)
    imsave(savefolder + sequence['name'] + '.png', im)

    views = ['visible', 'gt', 'rays_cobweb']
    for view in views:
        imgname = sequence['name'] + '_' + view + '_view_000.png'

        if count==5:
            imgname = imgname.replace('000', '002')

        if sequence['name'].startswith('4zy'):
            imgname = imgname.replace('000', '002')

        if sequence['name'].startswith('7bi'):
            imgname = imgname.replace('000', '001')

        if sequence['name'].startswith('7sm'):
            imgname = imgname.replace('000', '002')

        if sequence['name'].startswith('k4'):
            imgname = imgname.replace('000', '003')



        f.write(A + savefolder + imgname + '} ')

        if view == views[-1]:
            f.write('\\\\\n')
        else:
            f.write('& \n')

        # now copying the image into the folder
        shutil.copy(source_dir + imgname, savefolder)



f.close()