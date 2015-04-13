'''
make predictions for all of bigbird dataset
using my algorhtm
saves each prediction to disk
'''
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cPickle as pickle
import sys
import os
sys.path.append(os.path.expanduser("~/projects/shape_sharing/src/"))
from time import time
import scipy.misc  # for image saving
import shutil
import collections

import paths
import parameters

from common import voxlets
from common import scene

import sklearn.metrics

print "Loading model..."
with open(paths.model, 'rb') as f:
    model_without_implicit = pickle.load(f)
print "Done loading model..."

combine_renders = True
render_predictions = True
render_top_view = True
save_prediction_grids = True

# this overrides all other parameters. Means we don't botther with orables etc
only_prediction = False

class Voxlet(object):
    '''
    defining a class for voxlet parameters for these scenes, so we can adjust them...
    '''
    # setting some voxlet params here
    # NOTE BE VERY CAREFUL IF EDITING THESE
    tall_voxlets = True

    one_side_bins = 20
    shape = (one_side_bins, 2*one_side_bins, 2*one_side_bins)
    size = 0.0175 / 3.0  # edge size of a single voxel
    # centre is relative to the ijk origin at the bottom corner of the voxlet
    # z height of centre takes into account the origin offset
    actual_size = np.array(shape) * size
    centre = np.array((actual_size[0] * 0.5,
                       actual_size[1] * 0.25,
                       0.375+0.03))

    tall_voxlet_height = 0.375


model_without_implicit.voxlet_params = Voxlet

print "MAIN LOOP"
# Note: if parallelising, should either do here (at top level) or at the
# bottom level, where the predictions are accumulated (although this might be)
# better off being GPU...)
def process_sequence(sequence):

    sc = scene.Scene(parameters.RenderedVoxelGrid.mu,
        model_without_implicit.voxlet_params)
    sc.load_sequence(sequence, frame_nos=0, segment_with_gt=True,
        save_grids=False, load_implicit=False, voxel_normals='im_tsdf')
    # sc.santity_render(save_folder='/tmp/')

    test_type = 'oma'

    print "-> Reconstructing with oma forest"
    rec = voxlets.Reconstructer(
        reconstruction_type='kmeans_on_pca', combine_type='modal_vote')
    rec.set_scene(sc)
    rec.sample_points(parameters.VoxletPrediction.number_samples,
                      parameters.VoxletPrediction.sampling_grid_size)

    # Path where any renders will be saved to
    gen_renderpath = paths.voxlet_prediction_img_path % \
        (sequence['name'], '%s')
    print gen_renderpath

    print "-> Creating folder"
    fpath = paths.voxlet_prediction_folderpath % sequence['name']
    if not os.path.exists(fpath):
        os.makedirs(fpath)

    rec.initialise_output_grid(gt_grid=sc.gt_tsdf)
    rec.set_model(model_without_implicit)

    if render_top_view:
        print "-> Rendering top view"
        rec.plot_voxlet_top_view(savepath=gen_renderpath % 'top_view')
        print gen_renderpath % 'top_view'

    pred_voxlets = rec.fill_in_output_grid_oma(
        render_type=[], add_ground_plane=True,
        combine_segments_separately=False, feature_collapse_type='pca')
    pred_voxlets_exisiting = rec.keeping_existing

    if only_prediction:
        pred_voxlets.render_view(gen_renderpath % 'pred_voxlets')
        return

    combines = collections.OrderedDict()
    combines['input'] = {'name':'Input image'}
    combines['visible'] = {'name':'Visible surfaces', 'grid':sc.im_tsdf}
    combines['pred_voxlets'] = {'name':'Voxlets', 'grid':pred_voxlets}

    if render_predictions:
        print "-> Rendering"
        for name, dic in combines.iteritems():
            if name != 'input':
                dic['grid'].render_view(gen_renderpath % name)


    if save_prediction_grids:
        print "-> Saving prediction grids"
        with open('/tmp/combines.pkl', 'w') as f:
            pickle.dump(combines, f, protocol=pickle.HIGHEST_PROTOCOL)


    print "Done sequence %s" % sequence['name']


# need to import these *after* the pool helper has been defined
if parameters.multicore:
    import multiprocessing
    pool = multiprocessing.Pool(parameters.cores)
    mapper = pool.map
else:
    mapper = map


if __name__ == '__main__':

    # temp = [s for s in paths.RenderedData.test_sequence() if s['name'] == 'd2p8ae7t0xi81q3y_SEQ']
    # print temp
    tic = time()
    mapper(process_sequence, paths.test_sequence[1:])
    print "In total took %f s" % (time() - tic)


