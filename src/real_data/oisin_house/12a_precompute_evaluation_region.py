import matplotlib.pyplot as plt
import numpy as np
import cPickle as pickle
import scipy.io
import sys
import os
from time import time
import yaml
import system_setup
import collections
sys.path.append(os.path.expanduser("~/projects/shape_sharing/src/"))
from common import voxlets, scene, mesh


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
elif parameters['testing_data'] == 'nyu_cad_silberman':
    import nyu_cad_paths_silberman as paths
else:
    raise Exception('Unknown training data')


def get_nyu_within_walls(sequence_name, sc):
    '''
    returns a 3D mask which is one where the voxel is within the 3D bounds of
    the NYU Scene
    '''

    # load the mesh obj file
    scene_obj = paths.data_folder + '../binvox_with_walls/%s.obj'
    ms = mesh.Mesh()
    ms.load_from_obj(scene_obj % sequence_name)

    # finding the extents in real world space and idx space
    wall_xyz_in_world_space = ms.vertices[:, [0, 2, 1]]
    start = sc.gt_tsdf.world_to_idx(
        np.min(wall_xyz_in_world_space, axis=0)[None, :])[0]
    finish = sc.gt_tsdf.world_to_idx(
        np.max(wall_xyz_in_world_space, axis=0)[None, :])[0]

    # truncating to within the bounds
    start[start < 0] = 0
    max_shape = np.array(sc.gt_tsdf.V.shape)
    too_big = finish > max_shape
    finish[too_big] = max_shape[too_big]

    # creating the masks
    extra_mask_x = sc.gt_tsdf.V.copy().astype(np.int32) * 0
    extra_mask_y = sc.gt_tsdf.V.copy().astype(np.int32) * 0
    extra_mask_z = sc.gt_tsdf.V.copy().astype(np.int32) * 0

    extra_mask_x[start[0]:finish[0]] = 1
    extra_mask_y[start[1]:finish[1]] = 1
    extra_mask_z[start[2]:finish[2]] = 1

    extra_mask = np.logical_and.reduce(
        (extra_mask_x, extra_mask_y, extra_mask_z)).astype(np.int32)

    return extra_mask


def process_sequence(sequence):

    print "-> Loading ground truth", sequence['name']
    sys.stdout.flush()

    fpath = paths.prediction_folderpath % (
        parameters['batch_name'], sequence['name'])
    gt_scene = pickle.load(open(fpath + 'ground_truth.pkl'))

    if parameters['testing_data'] == 'nyu_cad' and parameters['evaluate_inside_room_only']:
        extra_mask = get_nyu_within_walls(sequence['name'], gt_scene)
    else:
        extra_mask = None

    evaluation_region = gt_scene.form_evaluation_region()

    # now save the evaluation region
    savepath = paths.evaluation_region_path % (
        parameters['batch_name'], sequence['name'])
    print "Saving to ", savepath
    # pickle.dump(evaluation_region, open(savepath, 'w'), -1)
    scipy.io.savemat(savepath, {'evaluation_region':evaluation_region})


# need to import these *after* the pool helper has been defined
if system_setup.multicore:
    import multiprocessing
    mapper = multiprocessing.Pool(4).map
else:
    mapper = map


if __name__ == '__main__':

    results = mapper(process_sequence, paths.test_data)
    yaml.dump(results, open('./nyu_cad/all_results.yaml', 'w'))
