'''
script to voxelise all the bigbird meshes
'''
import os
from subprocess import call
import sys

sys.path.append('/Users/Michael/projects/shape_sharing/src/structured_train/')
from thickness import paths


for modelname in paths.modelnames:

    out_path = paths.base_path + "bigbird_meshes/" + modelname + "/meshes/voxelised.vox"

    if os.path.exists(out_path) and os.path.getsize(out_path) > 1000:
        print "Skipping " + modelname
        continue

    mesh_path = paths.base_path + "bigbird_meshes/" + modelname + "/meshes/poisson.obj"

    f = open(out_path, "w")

    print mesh_path
    call(['./voxelcvml', mesh_path], stdout=f)

    print "Done " + modelname