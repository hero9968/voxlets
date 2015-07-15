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

import real_data_paths as paths
import real_params as parameters

from common import voxlets
from common import scene

import sklearn.metrics

print "Loading model..."


with open('/media/ssd/data/oisin_house/models_full_split_not_tall/models/oma_cobweb.pkl', 'rb') as f:
    model_short = pickle.load(f)

print model_short.pca.components_.shape
with open('/media/ssd/data/oisin_house/models_full_split_tall/models/oma_cobweb.pkl', 'rb') as f:
    model_tall = pickle.load(f)

print model_tall.pca.components_.shape

cobweb = True
combine_renders = True
render_predictions = True
render_top_view = True
save_prediction_grids = True

# this overrides all other parameters. Means we don't botther with orables etc
only_prediction = False


# model_without_implicit.voxlet_params = Voxlet

print "MAIN LOOP"
# Note: if parallelising, should either do here (at top level) or at the
# bottom level, where the predictions are accumulated (although this might be)
# better off being GPU...)
def process_sequence(sequence):

    sc = scene.Scene(parameters.mu, [])
    sc.load_sequence(sequence, frame_nos=0, segment_with_gt=False,
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

    scipy.misc.imsave(gen_renderpath % 'input', sc.im.rgb)
    plt.imshow(sc.im.depth)
    plt.axis('off')
    plt.savefig(gen_renderpath % 'input_depth')
    plt.close()

    plt.imshow(sc.im.mask)
    plt.axis('off')
    plt.plot(rec.sampled_idxs[:, 1], rec.sampled_idxs[:, 0], 'ro')
    plt.savefig(gen_renderpath % 'input_mask')
    plt.close()
    # scipy.misc.imsave(gen_renderpath % 'input', sc.im.depth)

    rec.initialise_output_grid(gt_grid=sc.gt_tsdf)
    rec.set_probability_model_one(0.95)
    rec.set_model([model_short, model_tall])

    if render_top_view:
        print "-> Rendering top view"
        rec.plot_voxlet_top_view(savepath=gen_renderpath % 'top_view')
        print gen_renderpath % 'top_view'

    pred_voxlets = rec.fill_in_output_grid_oma(
        render_type=[], add_ground_plane=True,
        combine_segments_separately=False, feature_collapse_type='pca', cobweb=cobweb)
    pred_voxlets_remove_excess = rec.remove_excess

    if only_prediction:
        pred_voxlets.render_view(gen_renderpath % 'pred_voxlets', ground_height=0.03, xy_centre=True)
        return

    combines = collections.OrderedDict()
    combines['input'] = {'name':'Input image'}
    combines['visible'] = {'name':'Visible surfaces', 'grid':sc.im_tsdf}
    combines['pred_voxlets'] = {'name':'Voxlets', 'grid':pred_voxlets}
    combines['pred_voxlets_remove_excess'] = {'name':'Voxlets remove excess', 'grid':pred_voxlets_remove_excess}

    if render_predictions:
        print "-> Rendering"
        for name, dic in combines.iteritems():
            if name != 'input':
                dic['grid'].render_view(gen_renderpath % name, ground_height=0.03, xy_centre=True)


    if save_prediction_grids:
        print "-> Saving prediction grids"
        with open('/tmp/combines.pkl', 'w') as f:
            pickle.dump(combines, f, protocol=pickle.HIGHEST_PROTOCOL)


    print "Done sequence %s" % sequence['name']


# need to import these *after* the pool helper has been defined
if False:
    # parameters.multicore:
    import multiprocessing
    pool = multiprocessing.Pool(parameters.cores)
    mapper = pool.map
else:
    mapper = map


if __name__ == '__main__':

    temp = [s for s in paths.test_data if s['name'] == '0416_[0]']
    print temp
    tic = time()
    mapper(process_sequence, temp)
    print "In total took %f s" % (time() - tic)


