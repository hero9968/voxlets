import numpy as np
from subprocess import call

import sys, os
sys.path.append('/Users/Michael/projects/shape_sharing/src/structured_train/')
from thickness import mesh
from thickness import paths

def isflipped(ms):
    # determines if a mesh is insideout or not

    # take a normal:
    n = ms.vertices.shape[0]
    print "n = " + str(n)

    means = []

    for tempidx in np.random.randint(0, n, 20):

        idx = int(tempidx)

        norm = ms.norms[idx]

        # dot all the other points onto this normal...
        points_offset = ms.vertices - ms.vertices[idx]
        dotP = np.dot(norm, points_offset.T)

        means.append(np.mean(dotP))

    return np.mean(np.array(means)) > 0


mlpath = '/Applications/meshlab.app/Contents/MacOS/meshlabserver'
scriptpath = '/Users/Michael/projects/shape_sharing/src/preprocess/reorientate_ply_normals/flip_faces.mlx'

def flipmesh(modelname):
    print "Reorientating " + modelname
    in_filename = paths.base_path + 'bigbird_meshes/' + modelname + '/meshes/poisson.ply'
    out_filename = in_filename #'/Users/Michael/Desktop/test.ply'
    call((mlpath, '-i', in_filename, '-o', out_filename, '-s', scriptpath))

for modelname in paths.modelnames:
    print modelname
    ms = mesh.BigbirdMesh()
    ms.load_bigbird(modelname)
    ms.compute_vertex_normals()

    if isflipped(ms):
        flipmesh(modelname)
