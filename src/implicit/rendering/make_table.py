import yaml
import os, shutil, glob
from scipy.misc import imread, imsave

# make and empty folder
savefolder = 'implicit_imgs/'
files = glob.glob(savefolder + '*')
for f in files:
    os.remove(f)

source_dir = '/media/ssd/data/oisin_house/implicit/renders/'

A = '\includegraphics[height=\\omaheight, clip=true, trim=20 30 30 5]{'


yaml_file = '/media/ssd/data/oisin_house/train_test/test.yaml'
test_data = yaml.load(open(yaml_file))

f = open('table.tex', 'w')

print "Warning - only using a subset of the data"
for count, sequence in enumerate(test_data[:20]):

    # input view image
    f.write(A + savefolder + sequence['name'] + '} & \n')

    # now copy the input view image to the output folder
    impath = sequence['folder'] + sequence['scene'] + '/frames/%05d.ppm' % sequence['frames'][0]
    im = imread(impath)
    imsave(savefolder + sequence['name'] + '.png', im)

    for view in ['visible', 'gt', 'zheng2', 'rays']:
        imgname = sequence['name'] + '_' + view + '_view_000.png'

        if count==5:
            imgname = imgname.replace('000', '002')

        f.write(A + savefolder + imgname + '} ')

        if view == 'pred_remove_excess':
            f.write('\\\\\n')
        else:
            f.write('& \n')

        # now copying the image into the folder
        shutil.copy(source_dir + imgname, savefolder)



f.close()