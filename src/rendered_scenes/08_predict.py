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

from common import paths
from common import parameters
from common import voxlets
from common import scene

import sklearn.metrics

print "Loading model..."
with open(paths.RenderedData.voxlets_path + '/models/oma.pkl', 'rb') as f:
    model_without_implicit = pickle.load(f)
print "Done loading model..."

combine_renders = True
render_predictions = True
render_top_view = False
save_prediction_grids = True
save_scores_to_yaml = True
copy_to_dropbox = True and paths.host_name == 'biryani'
base_dropbox_path = paths.RenderedData.new_dropbox_dir()

print "MAIN LOOP"
# Note: if parallelising, should either do here (at top level) or at the
# bottom level, where the predictions are accumulated (although this might be)
# better off being GPU...)
def process_sequence(sequence):

    sc = scene.Scene(parameters.RenderedVoxelGrid.mu,
        model_without_implicit.voxlet_params)
    sc.load_sequence(sequence, frame_nos=0, segment_with_gt=True,
        save_grids=False, load_implicit=True)
    sc.santity_render(save_folder='/tmp/')

    test_type = 'oma_implicit'

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

    if render_top_view:
        print "-> Rendering top view"
        rec.plot_voxlet_top_view(savepath=gen_renderpath % 'top_view')
        print gen_renderpath % 'top_view'

    rec.initialise_output_grid(gt_grid=sc.gt_tsdf)
    rec.set_model(model_without_implicit)
    pred_voxlets = rec.fill_in_output_grid_oma(
        render_type=[], add_ground_plane=True,
        combine_segments_separately=False, feature_collapse_type='pca')
    pred_voxlets_exisiting = rec.keeping_existing

    rec.initialise_output_grid(gt_grid=sc.gt_tsdf)
    full_oracle_voxlets = rec.fill_in_output_grid_oma(
        render_type=[],oracle='gt', add_ground_plane=True, feature_collapse_type='pca')
    # full_oracle_voxlets = rec.keeping_existing

    rec.initialise_output_grid(gt_grid=sc.gt_tsdf)
    oracle_voxlets = rec.fill_in_output_grid_oma(
        render_type=[],oracle='pca', add_ground_plane=True, feature_collapse_type='pca')
    oracle_voxlets_existing = rec.keeping_existing

    rec.initialise_output_grid(gt_grid=sc.gt_tsdf)
    nn_oracle_voxlets = rec.fill_in_output_grid_oma(
        render_type=[], oracle='nn', add_ground_plane=True, feature_collapse_type='pca')
    nn_oracle_voxlets_existing = rec.keeping_existing

    rec.initialise_output_grid(gt_grid=sc.gt_tsdf)
    greedy_oracle_voxlets = rec.fill_in_output_grid_oma(
        render_type=[], oracle='greedy_add', add_ground_plane=True, feature_collapse_type='pca')
    greedy_oracle_voxlets_existing = rec.keeping_existing

    combines = [
        ['Input image', 'input'],
        ['Ground truth', 'gt', sc.gt_tsdf],
        ['Visible surfaces', 'visible', sc.im_tsdf],
        ['Full oracle (OR1)', 'full_oracle_voxlets', full_oracle_voxlets],
        ['Oracle using PCA (OR2)', 'oracle_voxlets', oracle_voxlets],
        ['Oracle using NN (OR3)', 'nn_oracle_voxlets', nn_oracle_voxlets],
        ['Oracle using Greedy (OR4)', 'greedy_oracle_voxlets', greedy_oracle_voxlets],
        ['Implicit prediction', 'implicit', sc.implicit_tsdf],
        ['Voxlets', 'pred_voxlets', pred_voxlets]]
        # ['Voxlets + visible', 'pred_voxlets_exisiting', pred_voxlets_exisiting]]
        # ['Voxlets + visible, using implicit', 'pred_voxlets_implicit_exisiting', pred_voxlets_implicit_exisiting]]
        # ['Voxlets using implicit', 'pred_voxlets_implicit', pred_voxlets_implicit],

    if render_predictions:
        print "-> Rendering"
        for c in combines[1:]:
            c[2].render_view(gen_renderpath % c[1])

    print "-> Computing the score for each prediction"
    for c in combines[3:]:
        c.append(sc.evaluate_prediction(c[2].V))

    if save_prediction_grids:
        print "-> Saving prediction grids"
        with open('/tmp/combines.pkl', 'w') as f:
            pickle.dump(combines, f, protocol=pickle.HIGHEST_PROTOCOL)

    # must save the input view to the save folder
    scipy.misc.imsave(gen_renderpath % 'input', sc.im.rgb)

    if combine_renders:
        print "-> Combining renders"
        su, sv = 3, 4

        fig = plt.figure(figsize=(25, 10), dpi=1000)
        plt.subplots(su, sv)
        plt.subplots_adjust(left=0, bottom=0, right=0.95, top=0.95, wspace=0.05, hspace=0.05)

        for count, c in enumerate(combines):

            if count >= su*sv:
                raise Exception("Error! Final subplot is reserved for the ROC curve")

            plt.subplot(su, sv, count + 1)
            plt.imshow(scipy.misc.imread(gen_renderpath % c[1]))
            plt.axis('off')
            plt.title(c[0], fontsize=10)

            " Add to the roc plot, which is in the final subplot"
            if count >= 3:
                "Add the AUC"
                plt.text(0, 50, "AUC=%0.3f, prec=%0.3f, rec=%0.3f" % (c[3]['auc'], c[3]['precision'], c[3]['recall']),
                    fontsize=6, color='white')

                plt.subplot(su, sv, su*sv)
                plt.plot(c[3]['fpr'], c[3]['tpr'], label=c[0])
                plt.hold(1)
                plt.legend(prop={'size':6}, loc='lower right')

            if c[1] == 'input':
                plt.hold(1)
                plt.plot(rec.sampled_idxs[:, 1], rec.sampled_idxs[:, 0], 'r.', ms=2)

        fname = 'all_' + sequence['name']
        all_savename = gen_renderpath.replace('png', 'pdf') % fname
        plt.savefig(all_savename, dpi=400)

        # Saving to the dropbox...
        if copy_to_dropbox:
            shutil.copy(all_savename, dropbox_path)

    if save_scores_to_yaml:
        print "-> Writing scores to YAML"
        results_dict = {}
        for c in combines[3:]:
            test_key = c[1]
            results_dict[test_key] = \
                {'description': c[0],
                 'auc':         float(c[3]['auc']),
                 'precision':   float(c[3]['precision']),
                 'recall':      float(c[3]['recall'])}

        with open(fpath + 'scores.yaml', 'w') as f:
            f.write(yaml.dump(results_dict, default_flow_style=False))


    print "Done sequence %s" % sequence['name']


# need to import these *after* the pool helper has been defined
if False:  # parameters.multicore:
    import multiprocessing
    pool = multiprocessing.Pool(parameters.cores)
    mapper = pool.map
else:
    mapper = map


if __name__ == '__main__':

    # temp = [s for s in paths.RenderedData.test_sequence() if s['name'] == 'd2p8ae7t0xi81q3y_SEQ']
    # print temp
    tic = time()
    mapper(process_sequence, paths.RenderedData.test_sequence()[1:])
    print "In total took %f s" % (time() - tic)


