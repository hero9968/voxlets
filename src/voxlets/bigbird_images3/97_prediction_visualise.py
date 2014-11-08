
import numpy as np
import matplotlib.pyplot as plt 
import cPickle as pickle
import sys, os
sys.path.append(os.path.expanduser("~/projects/shape_sharing/src/"))
sys.path.append(os.path.expanduser("~/projects/depth_prediction/"))

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import voxel_utils as vu
from skimage import measure

from common import paths

# params
level = 0.0
vis_style = 'marching_cubes'
pad_vol = False
plot_it = True
 
def plot_tsdf(vol):
    '''
    function to plot a level set prediction to screen
    could also make this plot into different subplot from different angles etc
    '''
    if ((vol < level).sum() > 0) and ((vol > level).sum() > 0):
        if vis_style == 'voxels':
            verts, faces = vu.create_voxel_grid(vol, level)
        else:
            verts, faces = measure.marching_cubes(vol, level)

        if pad_vol:
            # need to remove padding effect
            # TODO check if this works
            verts = verts - (1.0/np.asarray(vol.shape))*10

        if plot_it:
            fig = plt.figure(figsize=(10, 12))
            ax = fig.add_subplot(111, projection='3d')

            # Fancy indexing: `verts[faces]` to generate a collection of triangles
            mesh = Poly3DCollection(verts[faces])
            ax.add_collection3d(mesh)

            ax.set_xlabel("x-axis")
            ax.set_ylabel("y-axis")
            ax.set_zlabel("z-axis")

            ax.set_xlim(0, vol.shape[0])
            ax.set_ylim(0, vol.shape[1])
            ax.set_zlim(0, vol.shape[2])

            plt.show()

        verts = verts / 10.0  # so its a reasonable scale for blender
        vu.write_obj(verts, faces, op_dir + str(ii)+ '_' + vis_style + '.obj')

    else:
        print 'warning could not do this shape - all ampty or all full'


for modelname in paths.test_names:
    print "Doing model " + modelname
    for this_view_idx in [0, 10, 20, 30, 40]:

        test_view = paths.views[this_view_idx]
        print "Doing view " + test_view

        loadpath = '/Users/Michael/projects/shape_sharing/data/voxlets/bigbird/troll_predictions/%s_%s.pkl' % (modelname, test_view)

        print "loading the data"
        f = open(loadpath, 'rb')
        D = pickle.load(f)
        f.close()

        med = D['medioid']
        modal = D['modal']
        GT = D['gt']

        plot_tsdf(GT)
        sdafd




