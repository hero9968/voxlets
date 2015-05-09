# a script to copy certain result images to a folder...

folder = '/media/ssd/data/rendered_arrangements/voxlets/predictions/for_monday_paper/'
to_copy = ['jpop9l44loazrg6q_SEQ', 'z5crp98v05ddx6z6_SEQ', 'eu33x4vsa3gmuk89_SEQ', 'fjgt8hqqtkqaasdv_SEQ', 'vdcoc4lc2qpm3qr0_SEQ']
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
