'''
script to voxelise all the bigbird meshes
'''
import os
from subprocess import call

base_path = '/Users/Michael/projects/shape_sharing/data/'
models_path = base_path + 'bigbird/models.txt'

f = open(models_path, 'r')
for ff in f:
    modelname = ff.strip()

    out_path = base_path + "bigbird_meshes/" + modelname + "/meshes/voxelised.txt"

    #if os.path.exists(out_path) and os.path.getsize(out_path) > 1000:
     #   print "Skipping " + modelname
      #  continue

    mesh_path = base_path + "bigbird_meshes/" + modelname + "/meshes/poisson.obj"

    f = open(out_path, "w")

    call(['./voxel', mesh_path], stdout=f)

    print "Done " + modelname
    #break