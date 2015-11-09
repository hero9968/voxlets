
import numpy as np
import cPickle as pickle
import sys
import os
import yaml
import system_setup
import scipy.misc
import matplotlib.pyplot as plt
import subprocess as sp
sys.path.append(os.path.expanduser("~/projects/shape_sharing/src/"))
from common import voxlets, scene, mesh

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


def process_sequence(sequence):

    if '1056' not in sequence['name']:
        return

    print "-> Loading ground truth", sequence['name']
    fpath = paths.prediction_folderpath % (parameters['batch_name'], sequence['name'])

    # Path where any renders will be saved to
    gen_renderpath = paths.voxlet_prediction_img_path % \
        (parameters['batch_name'], sequence['name'], '%s')

    print "-> Main renders"
    for test_params in parameters['tests']:

        print "Rendering ", test_params['name']

        print "Rendering without excess removed (could change this in future...)"
        prediction_savepath = fpath + test_params['name'] + '_average.pkl'

        if not os.path.exists(prediction_savepath):
            print "Cannot find!"
            continue

        prediction = pickle.load(open(prediction_savepath))

        if True:
            # adding ground height
            ground_height = 0.03
            height_voxels = float(ground_height) / float(prediction.vox_size)
            prediction.V[:, :, :height_voxels] = -10
            temp_slice = prediction.V[:, :, height_voxels]
            temp_slice[np.isnan(temp_slice)] = 10
            prediction.V[:, :, height_voxels] = temp_slice

        print "CountV", prediction.countV.shape
        print "SumV", prediction.sumV.shape
        print "V", prediction.V.shape
        ms = mesh.Mesh()
        ms.from_volume(prediction, 0)
        ms.remove_nan_vertices()

        print "Vertx shape", ms.vertices.shape


        # ok so now use countV as a confidence proxy
        # each voxel has a confidence, so each vertex in the mesh can have
        # that confidence as a colour
        # must then write to ply not obj file...
        temp = prediction.blank_copy()
        temp.V = prediction.countV
        idx = temp.world_to_idx(ms.vertices)
        countV_values = temp.get_idxs(idx).astype(np.float32)

        # convert these values to colours... (how?)
        countV_values -= countV_values.min()
        countV_values /= countV_values.max()  # normalise to [0, 1]
        countV_values += 0.3
        countV_values /= countV_values.max()  # normalise to [0, 1]


        cmap_values = plt.get_cmap('hot')(countV_values)[:, :3]
        cmap_values *= 255
        cmap_values = cmap_values.astype(np.uint8)
        print "cmap Values", cmap_values.shape

        if True:
            # doing xy_centre
            cen = prediction.origin + (np.array(
                prediction.V.shape) * prediction.vox_size) / 2.0
            ms.vertices[:, :2] -= cen[:2]
            ms.vertices[:, 2] -= 0.05

        # now writing ply file
        savepath = gen_renderpath % test_params['name'] + '_confidence'
        #ms.write_to_ply(savepath + '.ply', colours=cmap_values)
        ms.write_to_ply('/tmp/tmp.ply', colours=cmap_values)
        ms.write_to_obj('/tmp/tmp.obj')

        print "Rendering...",
        sys.stdout.flush()
        blend_path = os.path.expanduser('~/projects/shape_sharing/src/rendered_scenes/spinaround/spin_vertex_colours.blend')
        blend_py_path = os.path.expanduser('~/projects/shape_sharing/src/rendered_scenes/spinaround/blender_spinaround_frame_vertex_colours.py')
        subenv = os.environ.copy()
        subenv['BLENDERSAVEFILE'] = '/tmp/tmp' #savepath
        sp.call(['blender',
                 blend_path,
                 "-b", "-P",
                 blend_py_path],
                 env=subenv)
                 #stdout=open(os.devnull, 'w'),
                 #close_fds=True)

        sds
        #
        # savepath = (gen_renderpath % test_params['name']) + '_no_removal'
        # print "Saving to ", savepath
        # prediction.render_view(savepath,
        #     xy_centre=True, ground_height=ground_height, keep_obj=True)


# need to import these *after* the pool helper has been defined
if False:#system_setup.multicore:
    import multiprocessing
    mapper = multiprocessing.Pool(system_setup.testing_cores).map
else:
    mapper = map


if __name__ == '__main__':

    # print "WARNING - SMALL TEST DATA"
    # test_data = yaml.load(open('/media/ssd/data/oisin_house/train_test/test.yaml'))
    test_data = paths.test_data
    results = mapper(process_sequence, test_data)
