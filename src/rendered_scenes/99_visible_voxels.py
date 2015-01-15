
import sys, os
import numpy as np
#from mayavi import mlab
import scipy.io

sys.path.append(os.path.expanduser('~/projects/shape_sharing/src/'))
from common import voxel_data
from common import carving
from common import mesh

filepath = '/Users/Michael/projects/shape_sharing/data/rendered_arrangements/renders/5PWXVRSOV1/voxelgrid2.pkl'

vox = voxel_data.load_voxels(filepath)
print vox.vox_size

vis = carving.VisibleVoxels()
vis.set_voxel_grid(vox)
visible = vis.find_visible_voxels()

print visible.V.shape
print visible.V.dtype
print np.sum(visible.V<0)
print np.sum(visible.V==0)
print np.sum(visible.V==1)

ms = mesh.Mesh()
ms.from_volume(visible, level=0.5)

print ms.vertices
print ms.faces

ms.write_to_obj('./temp.obj')
#scipy.io.savemat('./visible.mat', dict(V=visible.astype(np.float)))
#mlab.contour3d(visible.astype(np.float), contours=[0.5], transparent=True)
#mlab.show()
