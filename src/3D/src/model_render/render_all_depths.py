import os
from subprocess import call

base_path = os.path.expanduser("~/projects/shape_sharing/data/3D/basis_models/")
models_list = base_path + 'databaseFull/fields/models.txt'

# choose here which rendering algorithm we want
if False:
	python_filename = 'getDepthSequence.py'
	savefolder = 'renders/'
else:
	python_filename = 'getDepthSequenceBackFace.py'
	savefolder = 'render_backface/'


f = open(models_list, 'r')

for idx, line in enumerate(f):

	modelname = line.strip()

	# check if the output files exist
	outfolder = base_path + savefolder + modelname

	if os.path.isdir(outfolder):
		if len([name for name in os.listdir(outfolder)]) == 42:
			continue
	else:
		os.mkdir(outfolder)

	print "Done " + str(idx) 
	call(["python", python_filename, modelname, "1", "1", "42"])