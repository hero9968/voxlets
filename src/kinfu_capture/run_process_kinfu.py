'''
python script to run kinfu large scale,
save the results to computer and move to suitable file
also converts pcd to ascii
'''

from subprocess import call
import os
import shutil

# set up paths
bin_path = '/home/michael/build/pcl/build/bin/'
kinfu_path = bin_path + 'pcl_kinfu_largeScale'
convert_path = bin_path + 'pcl_convert_pcd_ascii_binary'
mesh_convert_path = bin_path + 'pcl_kinfu_largeScale_mesh_output'

out_dir_template = "./saved_%d/"

# find a new folder to put all data in
filenum = 0
while(os.path.exists(out_dir_template % filenum)):
	filenum += 1

outdir = out_dir_template % filenum
os.mkdir(outdir)

# create folder where kinfu will save the frames
if not os.path.exists('frames/'):
	os.makedirs('frames/') # where the kinfu will save the frames
else:
	raise Exception("Frames dir already exists - move or delete!")

# run kinfu
call(kinfu_path)

# on exit, covnrt cloud to ascii and also to mesh
if os.path.exists('world.pcd'):
	call((convert_path, 'world.pcd', outdir + 'world_ascii.pcd', '0'))
	call((mesh_convert_path, 'world.pcd'))
else:
	raise Exception('Cannot find world pcd')

# move the mesh to the folder
if os.path.exists('world.pcd'):
	shutil.move('mesh_1.ply', outdir)
else:
	raise Exception('Cannot find world pcd')


# mv snapshots to save
if os.path.exists('./frames/'):
	shutil.move('./frames/', outdir)
else:
	raise Exception('Cannot find kinfu snapshots')

