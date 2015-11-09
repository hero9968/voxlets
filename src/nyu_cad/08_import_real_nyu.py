import scipy.io

base = "/media/michael/Seagate/internet_datasets/rgbd/nyu_dataset/"
savedir = "/home/michael/projects/shape_sharing/data/cleaned_3D/renders_yaml_format/renders/"

# just do for test data...
import sys, os
sys.path.append(os.path.expanduser('/home/michael/projects/shape_sharing/src/real_data/'))
from oisin_house import nyu_cad_paths_silberman as paths
from scipy.misc import imsave, imread
import numpy as np
import shutil

# for idx in range(1, 1450):
for sequence in paths.test_data:

	idx = int(sequence['name'].split('_')[0])
	#
	# if os.path.exists(savedir + sequence['name'] + '/images/rgb_real.png'):
	# 	print "Skipping number ", idx
	# 	continue

	print "Doing number ", idx

	loadpath = base + 'images_depth/depth_%06d.mat'
	D = scipy.io.loadmat(loadpath % idx)
	depth = D['imgDepthOrig']

	loadpath = base + 'images_rgb/rgb_%06d.mat'
	D = scipy.io.loadmat(loadpath % idx)
	rgb = D['imgRgbOrig']

	# save the depth and rgb to disk
	imgssavedir = savedir + sequence['name'] + '/images/'

	savepath = imgssavedir + 'rgb_real.png'
	imsave(savepath, rgb)

	savepath = imgssavedir + 'depth_real.png'
	imsave(savepath, depth)

	savepath = imgssavedir + 'depth_real.mat'
	scipy.io.savemat(savepath, {'depth':depth})

	# depth_image_path.replace('.png', '_structure.mat'
	A = '/media/michael/Seagate/internet_datasets/rgbd/nyu_dataset/labels_structure/labels_%06d.mat' % idx
	B = imgssavedir + 'structure.mat'
	shutil.copy(A, B)
