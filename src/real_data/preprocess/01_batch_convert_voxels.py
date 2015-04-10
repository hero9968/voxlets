import subprocess as sp
import real_data_paths as paths
import os

for sequence in paths.scenes:
    scene = sequence['folder'] + sequence['scene']
    print scene

    if not os.path.exists(scene + '/dump.obj'):
        print "Processing ", scene
        sp.call(['python', paths.converter_path, scene + '/dump.voxels'])
    else:
        print "Skipping ", scene