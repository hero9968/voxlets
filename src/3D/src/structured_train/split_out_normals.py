'''
Script to split normals out of the combined renders mat files!
'''

import os
import scipy.io
import loadsave
import paths

#

for modelname in loadsave.load_object_names('all'):

	model_folder = paths.base_path + 'normals/' + modelname + '/'
	
	if not os.path.exists(model_folder):
		print "Creating directory for " + modelname
		os.mkdir(model_folder)
	elif len([name for name in os.listdir(model_folder) if os.path.isfile(model_folder + name)]) == 42:
		print "Skipping " + modelname
		continue

	print "Processing..." + modelname

	combined_path = paths.base_path + 'combined_renders/' + modelname + '.mat'
	temp = scipy.io.loadmat(combined_path, squeeze_me=True)

	for view_idx in range(42):
		normals =  temp['renders'][view_idx][1]
		savepath = model_folder + '/norms_' + str(view_idx+1) + '.mat'
		scipy.io.savemat(savepath, dict(normals=normals))
		
