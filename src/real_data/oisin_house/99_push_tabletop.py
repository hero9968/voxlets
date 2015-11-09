# now pushing the tabletop results to prism space


# push results to prism space

import real_data_paths as paths
import os, sys
import shutil
import subprocess as sp
from scipy.misc import imread, imsave

sys.path.append(os.path.expanduser("~/projects/shape_sharing/src/"))
from common import mesh

parameters = {'batch_name': 'cvpr2016'}

zheng_path = '/media/michael/Seagate/phd_projects/volume_completion_data/data/oisin_house/implicit/models/zheng_2/predictions/'
savedir = '/home/michael/prism/data5/projects/depth_completion/cvpr2016/tabletop/'
# savedir = '/home/michael/Desktop/tmp_prism2/'

#
# def copy_flipped_obj(from_path, to_path):
# 	ms = mesh.Mesh()
# 	ms.load_from_obj(from_path)
# 	ms.vertices[:, 0] *= -1
# 	ms.write_to_obj(to_path)
#
#
# def copy_flipped_im(from_path, to_path):
# 	im = imread(from_path)
# 	print to_path
# 	if len(im.shape) == 3:
# 		print im.shape
# 		imsave(to_path + '.png', im[:, ::-1, :])
# 	else:
# 		print im.shape
# 		imsave(to_path + '.png', im[:, ::-1])


for sequence in paths.test_data:

	# print sequence['name']
	print sequence['name'],

	if not os.path.exists(savedir + sequence['name']):
		os.makedirs(savedir + sequence['name'])

	#############################################
	# doing zheng
	#############################################

	# doing the OBJ
	from_path_obj = zheng_path + sequence['name'] + '/prediction_render2.png.obj'
	to_path_obj = savedir + sequence['name'] + '/zheng.png.obj'

	if os.path.exists(from_path_obj):
		shutil.copy(from_path_obj, to_path_obj)
	else:
		print "Failed", from_path_obj

	#############################################
	# images
	#############################################

	save_path = savedir + sequence['name']

	gen_renderpath = '/media/michael/Seagate/phd_projects/volume_completion_data/data/oisin_house/predictions/cvpr2016/' + sequence['name']
	# gen_renderpath = gen_renderpath.replace('%s.png', '/images/')
	print gen_renderpath


	in_path = gen_renderpath + '/input.png'
	shutil.copy(in_path, save_path)

	in_path = gen_renderpath + '/input_depth.png'
	shutil.copy(in_path, save_path)

	# now do nyu input...

	for test_name in ['mean', 'medioid']:
	# 'short_and_tall_samples_no_segment',
			# 'ground_truth', 'visible', ]:

		gen_renderpath = paths.voxlet_prediction_img_path % \
			(parameters['batch_name'], sequence['name'], '%s')

		im_path = gen_renderpath % test_name
		im_dest_path = savedir + sequence['name'] + '/' + test_name + '.png'

		obj_path = im_path + '.obj'
		obj_dest_path = savedir + sequence['name'] + '/' + test_name + '.png.obj'
		# print "From ", im_path
		# print "To ", im_dest_path
		# print "From ", obj_path
		# print "To ", obj_dest_path

		if os.path.exists(im_path) and os.path.exists(obj_path):
			shutil.copy(im_path, im_dest_path)
			shutil.copy(obj_path, obj_dest_path)
		else:
			print "Failed", test_name, sequence['name']
			print im_path, obj_path



sp.call(['chmod', '-R', 'uga+rwx', savedir])
