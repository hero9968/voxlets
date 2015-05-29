import numpy as np
import scipy.io
from time import time
import yaml
import sys, os
sys.path.append(os.path.expanduser('~/projects/shape_sharing/src/'))
sys.path.append(os.path.expanduser('~/projects/shape_sharing/src/implicit/'))

from common import scene, voxel_data
import real_data_paths as paths

import matplotlib.pyplot as plt

import find_axes
# myimp = __import__('06_test_implicit_model')

parameters = yaml.load(open('../implicit_params.yaml'))

plot_segmentation = False
the_zheng_parameter = 2
modelname = 'zheng_' + str(the_zheng_parameter)
render = False

def process_sequence(sequence):

    print "Loading sequence %s" % sequence['name']

    # this is where to save the results...
    results_foldername = \
        paths.implicit_predictions_dir % (modelname, sequence['name'])
    print "Creating %s" % results_foldername
    if not os.path.exists(results_foldername):
        os.makedirs(results_foldername)

    print "Processing " + sequence['name']
    sc = scene.Scene(parameters['mu'], None)
    sc.load_sequence(
        sequence,
        segment_base=0.03,
        frame_nos=0,
        segment_with_gt=True,
        segment=True,
        save_grids=False)

    if plot_segmentation:
        plt.subplot(121)
        plt.imshow(sc.im.rgb)
        plt.subplot(122)
        plt.imshow(sc.gt_im_label)
        plt.savefig(results_foldername + 'segmentation.png')

    print "Doing zheng"
    pred_grid = find_axes.process_scene(sc, the_zheng_parameter)

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
            xy_centre=True, ground_height=0.03, keep_obj=True)
        sc.im_tsdf.render_view(results_foldername + 'visible_render.png',
            xy_centre=True, keep_obj=True)
        sc.gt_tsdf.render_view(results_foldername + 'gt_render.png',
            xy_centre=True, keep_obj=True)

    print "Evaluating"
    results = sc.evaluate_prediction(pred_grid.V)
    yaml.dump(results, open(results_foldername + 'eval.yaml', 'w'))


# need to import these *after* the pool helper has been defined
if True:
    import multiprocessing
    mapper = multiprocessing.Pool(6).map
else:
    mapper = map


if __name__ == '__main__':

    tic = time()
    print "DANGER - doing on train sequence"
    mapper(process_sequence, paths.test_data)
    print "In total took %f s" % (time() - tic)
