import numpy as np
import cPickle as pickle
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.expanduser("~/projects/shape_sharing/src/"))
from time import time
import yaml
import scipy.io
import scipy.misc

from common import voxlets, scene, mesh, images, voxel_data, carving, features

import system_setup

parameters_path = sys.argv[1]
parameters = yaml.load(open(parameters_path))
train_parameters = yaml.load(open(parameters_path.replace('testing', 'training')))



if parameters['testing_data'] == 'oisin_house':
    import real_data_paths as paths
elif parameters['testing_data'] == 'synthetic':
    import synthetic_paths as paths
elif parameters['testing_data'] == 'nyu_cad':
    import nyu_cad_paths as paths
else:
    raise Exception('Unknown training data')

def get_scene_pose(scene):
    with open(scene + '/scene_pose.yaml') as f:
        return yaml.load(f)


sequence = paths.test_data[0]
sequence['name'] = 'saved_00207_[536]'
sequence['scene'] = 'saved_00207'
sequence['frames'] = [0, 50, 100, 150, 200, 250, 300]
print sequence

# loading the video
# vid = images.RGBDVideo()
# scenepath = sequence['folder'] + sequence['scene']
# vid.load_from_yaml(scenepath + '/poses.yaml', frames = sequence['frames'])
# voxlet_params = train_parameters['voxlet_sizes']['short']
# sc = scene.Scene(train_parameters['mu'], voxlet_params)
# sc.load_sequence(sequence,
#     frame_nos = np.arange(10),
#     segment_with_gt=train_parameters['segment_with_gt'],
#     segment=train_parameters['segment_scene'])
# print sc.im.depth.shape

# load the scene parameters...
scene_pose = get_scene_pose(sequence['folder'] + sequence['scene'])
vgrid_size = np.array(scene_pose['size'])
voxel_size = train_parameters['voxel_size']
vgrid_shape = vgrid_size / voxel_size

# initialise voxel grid (could add helper function to make it explicit...?)
vox = voxel_data.WorldVoxels()
vox.V = np.zeros(vgrid_shape, np.uint8)
vox.set_voxel_size(voxel_size)
vox.set_origin(np.array([0, 0, 0]))

carver = carving.Fusion()
carver.set_voxel_grid(vox)
carver._set_up()
# prediction_grid = sc.gt_tsdf.blank_copy()

# we should load in a scene... perhaps without an image?
sc = scene.Scene(train_parameters['mu'], [])
sc.load_sequence(
    sequence, frame_nos=0, segment_with_gt=False, segment=False)
sc.sample_points(parameters['general_params']['number_samples'],
    nyu='nyu_cad' in parameters['testing_data'])
sc.im._clear_cache()
sc.im = []

vox_model_path = paths.voxlet_model_path
print [vox_model_path % name for name in parameters['general_params']['models_to_use']]
models = [pickle.load(open(vox_model_path % name))
          for name in parameters['general_params']['models_to_use']]

rec = voxlets.Reconstructer()
# rec.set_scene(sc)
# rec.initialise_output_grid(gt_grid=sc.gt_tsdf,
#     keep_explicit_count=parameters['default_reconstruction_params']['weight_predictions'])
rec.set_model(models)
rec.mu = train_parameters['mu']


# temp...
parameters['general_params']['number_samples'] = 100


for count, frame_num in enumerate(sequence['frames']):

    # setting up the scene with the current image...
    sc.current_kinfu = carver.accum.get_current_tsdf()

    sc.frame_data = sc._load_scene_data(
        sequence['folder'] + sequence['scene'], frame_num)

    sc.im = images.RGBDImage.load_from_dict(
        sequence['folder'] + sequence['scene'], sc.frame_data)

    norm_engine = features.Normals()
    sc.im.normals = norm_engine.compute_bilateral_normals(sc.im, stepsize=2)

    norm_nans = np.any(np.isnan(sc.im.get_world_normals()), axis=1)
    sc.im.mask = np.logical_and(
        ~norm_nans.reshape(sc.im.mask.shape),
        sc.im.mask)

    sc.sample_points(parameters['general_params']['number_samples'],
        nyu='nyu_cad' in parameters['testing_data'])

    # add this frame to voxel volume
    carver.integrate_image(
        sc.im, mu=train_parameters['mu'],
        filtering=False, measure_in_frustrum=False)

    vox = carver.accum.get_current_tsdf()
    visible = carver.visible_voxels

    # make predictions which match the current kinfu grid..
    # add these predictions into the prediction grid

    # forming th Reconstructer object and doing prediction
    rec.initialise_output_grid(gt_grid=sc.gt_tsdf,
        keep_explicit_count=parameters['default_reconstruction_params']['weight_predictions'])
    rec.set_model_probabilities(parameters['general_params']['model_probabilities'])
    rec.set_model(models)
    rec.set_scene(sc)
    prediction_grid = rec.fill_in_output_grid(
        scene_grid_for_comparison='current_kinfu',
        **parameters['default_reconstruction_params'])

    # now add the latest observations into the prediction grid
    to_update = ~np.isnan(sc.current_kinfu.V)
    print "Num nans", np.isnan(sc.current_kinfu.V).sum()
    prediction_grid.V[to_update] = sc.current_kinfu.V[to_update]

    # save the kinfu grid and the prediction grid
    # pickle.dump(open('/tmp/'))

    gen_renderpath = paths.kinfu_prediction_img_path % \
            (parameters['batch_name'], sequence['name'], '%s')

    print gen_renderpath[:-6]
    if not os.path.exists(gen_renderpath[:-6]):
        os.makedirs(gen_renderpath[:-6])

    # render view of predictoin
    savepath1 = gen_renderpath % ('prediction_%04d_%04d.png' % (count, frame_num))
    prediction_grid.render_view(savepath1,
        xy_centre=True, ground_height=0.03, keep_obj=True)

    # render view of kinfu
    savepath2 = gen_renderpath % ('kinfu_%04d_%04d.png' % (count, frame_num))
    sc.current_kinfu.render_view(savepath2,
        xy_centre=True, ground_height=0.03, keep_obj=True)

    savepath3 = gen_renderpath % ('input_%04d_%04d.png' % (count, frame_num))
    scipy.misc.imsave(savepath3, sc.im.rgb)

    # savepath4 = gen_renderpath % ('input_%04d_%04d.png' % (count, frame_num))
