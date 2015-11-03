import os
import shutil


start = "/home/michael/projects/shape_sharing/data/cleaned_3D/segmented_renders_no_walls/"
dest = "/home/michael/projects/shape_sharing/data/cleaned_3D/renders_yaml_format/renders/%s/images/segmented_01_0000.png"

fnames = os.listdir(start)

for fname in fnames:

	print fname
	newpath = dest % fname
	print newpath

	shutil.copy(start + fname + '/rgb.png', newpath)
