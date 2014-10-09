import os
from subprocess import call

base_path = os.path.expanduser("~/projects/shape_sharing/data/")
models_list = base_path + 'bigbird/models.txt'

# choose here which rendering algorithm we want
if False:
	python_filename = 'getDepthSequenceBB.py'
	savefolder = 'renders/'
else:
	python_filename = 'getDepthSequenceBackFaceBB.py'
	savefolder = 'render_backface/'


f = open(models_list, 'r')

for idx, line in enumerate(f):

	modelname = line.strip()

	# check if the output files exist
	outfolder = base_path + "bigbird_renders/" + savefolder + modelname

	if os.path.isdir(outfolder):
		if len([name for name in os.listdir(outfolder)]) == 75:
			print "Skipping " + modelname
			continue
	else:
		os.mkdir(outfolder)

	call(["python", python_filename, modelname])
	print "Done " + str(idx) 