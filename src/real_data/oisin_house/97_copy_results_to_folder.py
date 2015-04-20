# a script to copy certain result images to a folder...

folder = '/media/ssd/data/oisin_house/predictions/brand_new_short_voxlets'
to_copy = ['']
import os, sys,shutil

files = ['gt.png', 'pred_voxlets.png', 'visible.png', 'input.png']

savefolder ='/home/michael/Desktop/for_paper/'

for copy in to_copy:
    if not os.path.exists(folder + copy):
        continue
    for filename in files:

        start_path = folder + copy + '/' + filename
        end_path = savefolder + copy + '_' + filename

        shutil.copy(start_path, end_path)
