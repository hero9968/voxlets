import numpy as np
import cPickle as pickle
import sys
import os
sys.path.append(os.path.expanduser("~/projects/shape_sharing/src/"))
from time import time
import yaml
import scipy.io

from common import voxlets, scene, mesh, images, voxel_data, carving

import system_setup

if len(sys.argv) > 1:
    parameters_path = sys.argv[1]
else:
    parameters_path = './testing_params_nyu.yaml'
parameters = yaml.load(open(parameters_path))

if parameters['testing_data'] == 'oisin_house':
    import real_data_paths as paths
elif parameters['testing_data'] == 'synthetic':
    import synthetic_paths as paths
elif parameters['testing_data'] == 'nyu_cad':
    import nyu_cad_paths as paths
else:
    raise Exception('Unknown training data')


scene = 

# loading the video
vid = images.RGBDVideo()
vid.load_from_yaml(scene + '/poses.yaml')

# load the scene parameters...
scene_pose = get_scene_pose(scene)
vgrid_size = np.array(scene_pose['size'])
voxel_size = parameters['voxel_size']
vgrid_shape = vgrid_size / voxel_size

# initialise voxel grid (could add helper function to make it explicit...?)
vox = voxel_data.WorldVoxels()
vox.V = np.zeros(vgrid_shape, np.uint8)
vox.set_voxel_size(voxel_size)
vox.set_origin(np.array([0, 0, 0]))

# setting up the voxel carver
carver = carving.Fusion()
carver.set_voxel_grid(vox)

prediction_grid = ...

for frame_num, im in enumerate(vid.frames):

    # add this frame to voxel volume
    vox, visible = carver.integrate_image(
        im, mu=parameters['mu'], filtering=False, measure_in_frustrum=False)

    # make predictions which match the current kinfu grid..
    # add these predictions into the prediction grid
    rec.initialise_output_grid(prediction_grid)
    rec.set_model_probabilities(params['model_probabilities'])
    rec.set_model(models)
    sc.current_kinfu = carver.accum.get_current_tsdf()
    rec.set_scene(sc)
    prediction_grid = \
        rec.fill_in_output_grid(scene_grid_for_comparison='current_kinfu')

    # now add the latest observations into the prediction grid
    to_update = ~np.isnan(sc.current_kinfu.V)
    prediction_grid[to_update] = sc.current_kinfu.V[to_update]

    # save the kinfu grid and the prediction grid
    # pickle.dump(open('/tmp/'))

	gen_renderpath = paths.kinfu_prediction_img_path % \
	        (parameters['batch_name'], sequence['name'], '%s')

    # render view of predictoin
    savepath = gen_renderpath % ('prediction_%04d.png' % frame_num)
    prediction_grid.render_view(savepath,
        xy_centre=True, ground_height=0.03, keep_obj=True)

	# render view of kinfu
	savepath = gen_renderpath % ('kinfu_%04d.png' % frame_num)
	sc.current_kinfu.render_view(savepath,
        xy_centre=True, ground_height=0.03, keep_obj=True)
