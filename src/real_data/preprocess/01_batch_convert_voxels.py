import os
import subprocess as sp

data_folder = '/Users/Michael/projects/shape_sharing/data/desks/oisin_1/data/'
converter_path = '/Users/Michael/projects/InfiniTAM_Alt/convertor/voxels_to_ply.py'


scenes = [os.path.join(data_folder,o)
          for o in os.listdir(data_folder)
          if os.path.isdir(os.path.join(data_folder,o))]

for scene in scenes:

    if not os.path.exists(scene + '/dump.obj'):
        print "Processing ", scene
        sp.call(['python', converter_path, scene + '/dump.voxels'])
    else:
        print "Skipping ", scene

