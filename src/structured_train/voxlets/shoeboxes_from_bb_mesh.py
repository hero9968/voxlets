import numpy as np
import cPickle as pickle
import sys, os
sys.path.append('/Users/Michael/projects/shape_sharing/src/structured_train/')

from thickness import voxel_data
from thickness import paths
from thickness import mesh

'''
computes a shoebox for many points from each mesh
combines them all together
will then do analysis on them
'''

#TODO - should ignore points that are facing downwards...

# params for the shoeboxes
shoebox_gridsize = (20, 20, 20)
shoebox_p_from_origin = np.array((0.1, 0.05, 0.1))
shoebox_voxelsize = 0.2/20.0

number_points_from_each_model = 250

# all the shoeboxes are stored in a dictionary
all_shoeboxes = {}

for modelname in paths.modelnames:

    # loading this mesh and computing the vertex normals
    ms = mesh.BigbirdMesh()
    ms.load_bigbird(modelname)
    ms.compute_vertex_normals()

    # loading the voxel grid for this model
    vgrid = voxel_data.BigBirdVoxels()
    vgrid.load_bigbird(modelname)

    model_shoeboxes = []

    # For a selection of the normals, create a shoebox
    all_point_idxs = range(0, ms.vertices.shape[0])
    for point_idx in np.random.choice(all_point_idxs, number_points_from_each_model, replace=False):

        # create single shoebox
        shoebox = voxel_data.ShoeBox(shoebox_gridsize) # grid size
        shoebox.set_p_from_grid_origin(shoebox_p_from_origin) # metres
        shoebox.set_voxel_size(shoebox_voxelsize) # metres

        shoebox.initialise_from_point_and_normal(ms.vertices[point_idx], 
                                                 ms.norms[point_idx], 
                                                 np.array([0, 0, 1]))

        # convert the indices to world xyz space
        shoebox_xyz_in_world = shoebox.world_meshgrid()
        shoebox_xyx_in_world_idx, valid = vgrid.world_to_idx(shoebox_xyz_in_world, True)

        # fill in the shoebox voxels
        idxs = shoebox_xyx_in_world_idx[valid, :]
        occupied_values = vgrid.extract_from_indices(idxs)
        shoebox.set_indicated_voxels(valid, occupied_values)

        model_shoeboxes.append(shoebox)

        print(str(point_idx))

    # add to the dictionary
    all_shoeboxes[modelname] = model_shoeboxes
    print "Done model " + modelname

# saving all the shoeboxes
save_path = paths.base_path + "voxlets/" + "all_shoeboxes.pkl"
f = open(save_path, 'wb')
pickle.dump(all_shoeboxes, f, protocol=2)
f.close()

