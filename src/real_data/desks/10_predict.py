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
import yaml
import shutil
import collections

import real_data_paths as paths
import real_params as parameters

from common import voxlets
from common import scene

import sklearn.metrics

print "Loading model..."
with open(paths.RenderedData.voxlet_model_oma_path, 'rb') as f:
    model_without_implicit = pickle.load(f)
print "Done loading model..."

combine_renders = True
render_predictions = True
render_top_view = True
save_prediction_grids = True
save_scores_to_yaml = True

# this overrides all other parameters. Means we don't botther with orables etc
only_prediction = False

print "MAIN LOOP"
# Note: if parallelising, should either do here (at top level) or at the
# bottom level, where the predictions are accumulated (although this might be)
# better off being GPU...)
def process_sequence(sequence):

    sc = scene.Scene(parameters.mu,
        model_without_implicit.voxlet_params)
    sc.load_sequence(sequence, frame_nos=0, segment_with_gt=True,
        save_grids=False, load_implicit=False, voxel_normals='gt_tsdf')
    # sc.santity_render(save_folder='/tmp/')

    test_type = 'oma_choose_nearest'

    print "-> Reconstructing with oma forest"
    rec = voxlets.Reconstructer(
        reconstruction_type='kmeans_on_pca', combine_type='modal_vote')
    rec.set_scene(sc)
    rec.sample_points(parameters.VoxletPrediction.number_samples,
                      parameters.VoxletPrediction.sampling_grid_size)

    # Path where any renders will be saved to
    gen_renderpath = paths.RenderedData.voxlet_prediction_img_path % \
        (test_type, sequence['name'], '%s')

    print "-> Creating folder"
    fpath = paths.RenderedData.voxlet_prediction_folderpath % \
        (test_type, sequence['name'])
    if not os.path.exists(fpath):
        os.makedirs(fpath)

    rec.initialise_output_grid(gt_grid=sc.gt_tsdf)
    rec.set_model(model_without_implicit)
    pred_voxlets = rec.fill_in_output_grid_oma(
        render_type=[], add_ground_plane=True,
        combine_segments_separately=False, feature_collapse_type='pca')
    pred_voxlets_exisiting = rec.keeping_existing

    if only_prediction:
        pred_voxlets.render_view(gen_renderpath % 'pred_voxlets')
        return

    if render_top_view:
        print "-> Rendering top view"
        rec.plot_voxlet_top_view(savepath=gen_renderpath % 'top_view')
        print gen_renderpath % 'top_view'


    combines = collections.OrderedDict()
    combines['input'] = {'name':'Input image'}
    combines['gt'] = {'name':'Ground truth', 'grid':sc.gt_tsdf}
    combines['visible'] = {'name':'Visible surfaces', 'grid':sc.im_tsdf}
    combines['pred_voxlets'] = {'name':'Voxlets', 'grid':pred_voxlets}
    combines['pred_voxlets_exisiting'] = {'name':'Voxlets existing', 'grid':pred_voxlets_exisiting}


    if render_predictions:
        print "-> Rendering"
        for name, dic in combines.iteritems():
            if name != 'input':
                dic['grid'].render_view(gen_renderpath % name)

    print "-> Computing the score for each prediction"
    for name, dic in combines.iteritems():
        if name != 'input':
            dic['results'] = sc.evaluate_prediction(dic['grid'].V)

    if save_prediction_grids:
        print "-> Saving prediction grids"
        with open(fpath + 'all_grids.pkl', 'w') as f:
            pickle.dump(combines, f, protocol=pickle.HIGHEST_PROTOCOL)

    # must save the input view to the save folder
    scipy.misc.imsave(gen_renderpath % 'input', sc.im.rgb)

    if combine_renders:
        print "-> Combining renders"
        su, sv = 2, 3

        fig = plt.figure(figsize=(25, 10), dpi=1000)
        plt.subplots(su, sv)
        plt.subplots_adjust(left=0, bottom=0, right=0.95, top=0.95, wspace=0.05, hspace=0.05)

        for count, (name, dic) in enumerate(combines.iteritems()):

            if count >= su*sv:
                raise Exception("Error! Final subplot is reserved for the ROC curve")

            plt.subplot(su, sv, count + 1)
            plt.imshow(scipy.misc.imread(gen_renderpath % name))
            plt.axis('off')
            plt.title(dic['name'], fontsize=10)

            " Add to the roc plot, which is in the final subplot"
            if name != 'input':
                "Add the AUC"
                plt.text(0, 50, "AUC=%0.3f, prec=%0.3f, rec=%0.3f" % \
                    (dic['results']['auc'],
                     dic['results']['precision'],
                     dic['results']['recall']),
                    fontsize=6, color='white')

                plt.subplot(su, sv, su*sv)
                plt.plot(dic['results']['fpr'], dic['results']['tpr'], label=dic['name'])
                plt.hold(1)
                plt.legend(prop={'size':6}, loc='lower right')

            else:
                plt.hold(1)
                plt.plot(rec.sampled_idxs[:, 1], rec.sampled_idxs[:, 0], 'r.', ms=2)

        fname = 'all_' + sequence['name']
        all_savename = gen_renderpath.replace('png', 'pdf') % fname
        plt.savefig(all_savename, dpi=400)

    if save_scores_to_yaml:
        print "-> Writing scores to YAML"
        results_dict = {}
        for name, dic in combines.iteritems():
            if name != 'input':
                test_key = name
                results_dict[test_key] = \
                    {'description': dic['name'],
                     'auc':         float(dic['results']['auc']),
                     'precision':   float(dic['results']['precision']),
                     'recall':      float(dic['results']['recall'])}

        with open(fpath + 'scores.yaml', 'w') as f:
            f.write(yaml.dump(results_dict, default_flow_style=False))


    print "Done sequence %s" % sequence['name']


# need to import these *after* the pool helper has been defined
if True:
    # parameters.multicore:
    import multiprocessing
    pool = multiprocessing.Pool(parameters.cores)
    mapper = pool.map
else:
    mapper = map


if __name__ == '__main__':

    # temp = [s for s in paths.RenderedData.test_sequence() if s['name'] == 'd2p8ae7t0xi81q3y_SEQ']
    # print temp
    tic = time()
    mapper(process_sequence, paths.test_data)
    # [:48], chunksize=1)
    print "In total took %f s" % (time() - tic)


