# covnerting the baseline obj files into 'our' obj format

# let's set it up to render the ones we want
import subprocess as sp
import os, sys
sys.path.append('../../src/')
from common import voxel_data
import binvox_rw
import numpy as np

to_render = [1,32,195,564,591,620,520,698]

endings = ['lin', 'gcpr']

# ip_dir = '/home/michael/prism/data5/projects/depth_completion/cvpr2016/nyu/Geiger2015GCPR/'
ip_dir = ("/media/michael/Seagate/phd_projects/"
    "volume_completion_data/data/nyu_cad/baselines/")
op_dir = '/home/michael/prism/data5/projects/depth_completion/cvpr2016/nyu/from_cad/'

for idx in to_render:

    for ending in endings:

        print idx, ending
        sys.stdout.flush()

        ip_path = ip_dir + str(idx) + "_" + ending + ".obj"
        bvox_path = ip_dir + str(idx) + "_" + ending + ".binvox"

        # convert to binvox
        if not os.path.exists(bvox_path):
            sp.call(['binvox', ip_path])

        with open(bvox_path, 'r') as f:
            bvox = binvox_rw.read_as_3d_array(f)

        # convert binvox to voxelgrid
        vgrid = voxel_data.WorldVoxels()
        vgrid.set_origin(bvox.translate)
        vgrid.set_voxel_size(bvox.scale / float(bvox.data.shape[0]))
        vgrid.V = bvox.data.astype(np.float16)
        vgrid.V = vgrid.compute_tsdf(0.1)

        tempV = vgrid.V.copy().transpose((0, 2, 1))[:, :, :]
        vgrid.V = tempV.astype(np.float16)
        vgrid.origin = vgrid.origin[[0, 2, 1]]
        #
        # with open(new_dir + foldername + '/ground_truth_tsdf.pkl', 'w') as f:
        #     pickle.dump(vgrid, f, -1)

        # now render this...
        # savepath = '/home/michael/Desktop/baselines_voxelised/' + \
            # str(idx) + "_" + ending + "_voxelised.png"
        savepath = op_dir + mapper[idx] + '/' + ending + '.png'

        vgrid.render_view(savepath, xy_centre=False, ground_height=0.0,
            keep_obj=True, actually_render=False, flip=True)
