
import sys, os
import numpy as np
#from mayavi import mlab
import scipy.io

sys.path.append(os.path.expanduser('~/projects/shape_sharing/src/'))
from common import voxel_data
from common import carving
from common import mesh

filepath = '/Users/Michael/projects/shape_sharing/data/rendered_arrangements/sequences/vsup94if8oqlm2mi/visible_voxels.pkl'

vox = voxel_data.load_voxels(filepath)
vox.V = vox.V.astype(np.float)
print vox.vox_size
print np.min(vox.V)
print np.max(vox.V)

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

print "Saving obj"
ms.write_to_obj('./temp.obj')
#scipy.io.savemat('./visible.mat', dict(V=visible.astype(np.float)))
mlab.contour3d(vox.V.astype(np.float), contours=[0.5], transparent=True)
mlab.show()
