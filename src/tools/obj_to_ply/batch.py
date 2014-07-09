# batch converts obj files to ply files, using the binary obj_to_ply
# could in general be adapted to do batch operations on many types of file

import os
from subprocess import call

obj_path = '../data/'
ply_path = '../ply/'

all_files = os.listdir(obj_path)

for f in all_files:
	obj_in_file = obj_path + f

	ply_out_file = ply_path + f[:-4] + '.ply'

	to_run = "obj_to_ply < %s > %s" % (obj_in_file, ply_out_file)
	
	os.system(to_run)
	print to_run
