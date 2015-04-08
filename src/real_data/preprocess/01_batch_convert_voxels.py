import subprocess as sp
import real_data_paths as paths
import os

for scene in paths.scenes:

    if not os.path.exists(scene + '/dump.obj'):
        print "Processing ", scene
        sp.call(['python', converter_path, scene + '/dump.voxels'])
    else:
        print "Skipping ", scene