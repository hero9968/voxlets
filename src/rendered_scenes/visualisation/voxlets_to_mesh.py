'''
By OMA originally, adapted by MDF
'''
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import voxel_utils as vu
from skimage import measure
import pickle

import sys
import os
sys.path.append(os.path.expanduser('~/projects/shape_sharing/src/'))
from common import paths
from common import parameters

plt.close('all')
save_dir = paths.RenderedData.voxlets_dictionary_path + '/visualisation/kmeans/'
kmeans_savepath = paths.RenderedData.voxlets_dictionary_path + 'shoeboxes_kmean.pkl'
with open(kmeans_savepath, 'rb') as f:
    clusters = pickle.load(f)
level = 0.0

plot_it = False
save_to_disk = True
pad_vol = False
vis_style = 'marching_cubes'  # 'voxels' or 'marching_cubes'

print clusters.cluster_centers_.shape

for count, c in enumerate(clusters.cluster_centers_[:150]):
    print count
    vol = c.reshape(parameters.Voxlet.shape)

    # place it inside larger volume
    if pad_vol:
        vol_big = np.ones((vol.shape[0]+2, vol.shape[1]+2, vol.shape[2]+2))*vol.max()
        vol_big[1:-1, 1:-1, 1:-1] = vol
        vol = vol_big

    if ((vol < level).sum() > 0) and ((vol > level).sum() > 0):
        if vis_style == 'voxels':
            verts, faces = vu.create_voxel_grid(vol, level)
        else:
            vol[:, :, -2:] = parameters.RenderedVoxelGrid.mu
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

        print vol.shape
        
        verts *= parameters.Voxlet.size
        verts *= 10.0  # so its a reasonable scale for blender
        print verts.min(axis=0), verts.max(axis=0)
        vu.write_obj(verts, faces, save_dir + str(count)+ '_' + vis_style + '.obj')

    else:
        if (vol < level).sum() > 0:
            print 'warning could not do this shape - all full'
        else:
            print 'warning could not do this shape - all empty'
