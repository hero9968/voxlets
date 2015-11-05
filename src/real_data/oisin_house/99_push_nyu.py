# push results to prism space

import nyu_cad_paths_silberman as paths
import os, sys
import shutil
import subprocess as sp
from scipy.misc import imread, imsave

sys.path.append(os.path.expanduser("~/projects/shape_sharing/src/"))
from common import mesh

parameters = {'batch_name': 'nyu_cad_silberman'}

zheng_path = paths.data_folder_alt + '/implicit/models/zheng_2/predictions/%s/prediction_render.png'
savedir = '/home/michael/prism/data5/projects/depth_completion/cvpr2016/nyu/from_cad/'
# savedir = '/home/michael/Desktop/tmp_prism/'

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

	# doing zheng image:
	from_path = zheng_path % sequence['name']
	to_path = savedir + sequence['name'] + '/zheng.png'

	# doing the OBJ
	from_path_obj = zheng_path % sequence['name'] + '.obj'
	to_path_obj = savedir + sequence['name'] + '/zheng.png.obj'

	if os.path.exists(from_path):
		shutil.copy(from_path, to_path)
		shutil.copy(from_path_obj, to_path_obj)
	else:
		print "Failed", from_path

	#############################################
	# images
	#############################################

	save_path = savedir + sequence['name']

	gen_renderpath = '/home/michael/projects/shape_sharing/data/cleaned_3D/renders_yaml_format/renders/' + sequence['name'] + '/images/'
	# gen_renderpath = gen_renderpath.replace('%s.png', '/images/')
	print gen_renderpath


	in_path = gen_renderpath + 'depth.png'
	shutil.copy(in_path, save_path)

	in_path = gen_renderpath + 'rgb.png'
	shutil.copy(in_path, save_path)

	in_path = gen_renderpath + 'rgb_real.png'
	shutil.copy(in_path, save_path)

	# now do nyu input...

	for test_name in ['short_tall_samples_0.02_pointwise',
			'ground_truth', 'visible']:

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



sp.call(['chmod', '-R', 'uga+rwx', '/home/michael/prism/data5/projects/depth_completion/cvpr2016/'])
