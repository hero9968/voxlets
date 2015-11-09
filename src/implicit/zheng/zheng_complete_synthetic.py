import numpy as np
import scipy.io
from time import time
import yaml
import sys, os
import scipy.misc
import scipy.io
from scipy.io import loadmat

sys.path.append(os.path.expanduser('~/projects/shape_sharing/src/'))
sys.path.append(os.path.expanduser('~/projects/shape_sharing/src/implicit/'))

synth = False

from common import scene, voxel_data

if synth:
    import synthetic_paths as paths
    parameters = yaml.load(open('../implicit_params.yaml'))
else:
    import nyu_cad_paths_silberman as paths
    parameters = yaml.load(open('../../real_data/oisin_house/training_params_nyu_silberman.yaml'))

######################
######################
do_original = True
skip_done = False
######################
######################

import matplotlib.pyplot as plt

import find_axes
# myimp = __import__('06_test_implicit_model')

plot_segmentation = False
the_zheng_parameter = 2
if do_original:
    modelname = 'zheng_' + str(the_zheng_parameter) + '_real'
else:
    modelname = 'zheng_' + str(the_zheng_parameter)

render = True

def process_sequence(sequence):
    # if sequence['name'] != '285_classroom_0001_2':
    #     return

    print "Loading sequence %s" % sequence['name']

    evaluation_region_loadpath = paths.evaluation_region_path % (
        parameters['batch_name'], sequence['name'])

    if not os.path.exists(evaluation_region_loadpath):
        print "Counld not find evaluation region - skipping"
        print evaluation_region_loadpath
        sds
        return
    # this is where to save the results...
    results_foldername = \
        paths.implicit_predictions_dir % (modelname, sequence['name'])

    if os.path.exists(results_foldername + 'prediction.mat') and skip_done:
        print "Already done - skipping"
        return

    print "Creating %s" % results_foldername
    if not os.path.exists(results_foldername):
        os.makedirs(results_foldername)

    if os.path.exists(results_foldername + 'eval.yaml') and skip_done:
        print "Skipping ", sequence['name']
        return

    print "Processing " + sequence['name']
    sc = scene.Scene(parameters['mu'], None)
    sc.load_sequence(
        sequence,
        segment_base=0.00,
        frame_nos=0,
        segment_with_gt=False,
        segment=False,
        save_grids=False,
        original_nyu=do_original)
    print sequence

    if do_original:
        # load from nyu
        p1 = '/media/michael/Seagate/internet_datasets/rgbd/nyu_dataset/labels_objects/'
        p2 = '/media/michael/Seagate/internet_datasets/rgbd/nyu_dataset/labels_instances/'
        p3 = '/media/michael/Seagate/internet_datasets/rgbd/nyu_dataset/labels_structure/'
        
        sc_idx = int(sequence['name'].split('_')[0])
        instance = loadmat(
            p2 + 'labels_%06d.mat' % sc_idx)['imgInstanceLabelsOrig']
        obj = loadmat(
            p1 + 'labels_%06d.mat' % sc_idx)['imgObjectLabelsOrig']
        structure = loadmat(
            p3 + 'labels_%06d.mat' % sc_idx)['imgStructureLabelsOrig']

        to_use = (structure >= 3)

        all_labels = 0 * obj
        current_lab = 1
        for obj_label in np.unique(obj[to_use]):
            this_objs = obj == obj_label
            these_instance_labels = instance[this_objs]

            for this_instance in np.unique(these_instance_labels):
                to_update = np.logical_and(
                    instance==this_instance, obj==obj_label)
                all_labels[to_update] = current_lab
                current_lab += 1
        sc.gt_im_label = all_labels.astype(np.int32)
        skip_segment=0
    else:
        fpath = sequence['folder'] + sequence['scene'] + \
            '/images/segmented_' + sc.frame_data['id'] + '.png'

        sc.gt_im_label = scipy.misc.imread(fpath)[:, :, 0]
        sc.gt_im_label[sc.gt_im_label==254] = 255
        skip_segment=255
    print np.unique(sc.gt_im_label)

    if plot_segmentation:
        plt.subplot(121)
        plt.imshow(sc.im.rgb)
        plt.subplot(122)
        plt.imshow(sc.gt_im_label)
        plt.savefig(results_foldername + 'segmentation.png')

    print "Doing zheng"
    pred_grid = find_axes.process_scene(
        sc, the_zheng_parameter, skip_segment=skip_segment, resize=0.5)

    pred_grid.V = pred_grid.V.astype(np.float32)

    print "Converting to a tsdf equivalent"
    pred_grid.V[pred_grid.V > 0] = -1
    pred_grid.V[pred_grid.V == 0] = 1

    print (pred_grid.V < 0).sum()
    print (pred_grid.V > 0).sum()
    print (pred_grid.V == 0).sum()

    print "Saving result to disk"
    pred_grid.save(results_foldername + 'prediction.pkl')
    scipy.io.savemat(
        results_foldername + 'prediction.mat',
        dict(gt=sc.gt_tsdf.V, pred=pred_grid.V),
        do_compression=True)

    if render:
        print "Doing the rendering"
        pred_grid.render_view(results_foldername + 'prediction_render.png',
            xy_centre=True, ground_height=0.03, keep_obj=True, actually_render=False)
        sc.im_tsdf.render_view(results_foldername + 'visible_render.png',
            xy_centre=True, keep_obj=True, actually_render=False)
        sc.gt_tsdf.render_view(results_foldername + 'gt_render.png',
            xy_centre=True, keep_obj=True, actually_render=False)

    print "Evaluating"
    evaluation_region = scipy.io.loadmat(
        evaluation_region_loadpath)['evaluation_region'] > 0

    results = sc.evaluate_prediction(
        pred_grid.V, voxels_to_evaluate=evaluation_region)
    yaml.dump(results, open(results_foldername + 'eval.yaml', 'w'))


# need to import these *after* the pool helper has been defined
if 1:
    import multiprocessing
    mapper = multiprocessing.Pool(8).map
else:
    mapper = map


if __name__ == '__main__':

    tic = time()
    # print "DANGER - doing on train sequence"
    mapper(process_sequence, paths.test_data)
    print "In total took %f s" % (time() - tic)
