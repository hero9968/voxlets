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

import paths
import parameters

from common import voxlets
from common import scene

import sklearn.metrics

print "Loading model..."
with open(paths.RenderedData.voxlets_path + '/models/oma.pkl', 'rb') as f:
    model_without_implicit = pickle.load(f)
print "Done loading model..."

do_oracles = True
combine_renders = True
render_predictions = True
render_top_view = True
save_prediction_grids = True
save_scores_to_yaml = True
copy_to_dropbox = False and paths.host_name == 'biryani'
dropbox_path = paths.RenderedData.new_dropbox_dir()
print dropbox_path

# this overrides all other parameters. Means we don't botther with orables etc
only_prediction = False

test_type = 'true_greedy_oracles_pca400'

def save_render_assess(dic, sc):
    '''
    does whatever is required in terms of saving results, rendering them etc
    '''

    # Path where any renders will be saved to
    gen_renderpath = paths.RenderedData.voxlet_prediction_img_path % \
        (test_type, sc.sequence['name'], '%s')

    if render_predictions:
        dic['grid'].render_view(gen_renderpath % dic['desc'])

    dic['results'] = sc.evaluate_prediction(dic['grid'].V)

    # finally removing the grid
    dic['grid'] = []

    return dic


print "MAIN LOOP"
# Note: if parallelising, should either do here (at top level) or at the
# bottom level, where the predictions are accumulated (although this might be)
# better off being GPU...)
def process_sequence(sequence):

    sc = scene.Scene(parameters.RenderedVoxelGrid.mu,
        model_without_implicit.voxlet_params)
    sc.load_sequence(sequence, frame_nos=0, segment_with_gt=True,
        save_grids=False, load_implicit=False)
    # sc.santity_render(save_folder='/tmp/')

    # Path where any renders will be saved to
    gen_renderpath = paths.RenderedData.voxlet_prediction_img_path % \
        (test_type, sc.sequence['name'], '%s')

    print "-> Reconstructing with oma forest"
    rec = voxlets.Reconstructer(
        reconstruction_type='kmeans_on_pca', combine_type='modal_vote')
    rec.set_scene(sc)
    rec.sample_points(parameters.VoxletPrediction.number_samples,
                      parameters.VoxletPrediction.sampling_grid_size)

    print "-> Creating folder"
    fpath = paths.RenderedData.voxlet_prediction_folderpath % \
        (test_type, sequence['name'])
    if not os.path.exists(fpath):
        os.makedirs(fpath)

    rec.set_model(model_without_implicit)

    combines = collections.OrderedDict()
    combines['input'] = {'name':'Input image'}
    combines['gt'] = save_render_assess({'name':'Ground truth', 'grid':sc.gt_tsdf, 'desc':'gt'}, sc)
    combines['visible'] = save_render_assess({'name':'Visible surfaces', 'grid':sc.im_tsdf,'desc':'visible'}, sc)

    print '''DOING REAL PREDICTION'''

    rec.initialise_output_grid(gt_grid=sc.gt_tsdf)
    pred_voxlets = rec.fill_in_output_grid_oma(
        render_type=[], add_ground_plane=True,
        combine_segments_separately=False, feature_collapse_type='pca', use_binary=parameters.use_binary)
    combines['pred_voxlets'] = save_render_assess({'name':'Voxlets', 'grid':pred_voxlets, 'desc':'pred_voxlets'}, sc)
    pred_voxlets = None


    rec.initialise_output_grid(gt_grid=sc.gt_tsdf)
    rec.set_model(model_without_implicit)
    pred_voxlets_medioid = rec.fill_in_output_grid_oma(
        add_ground_plane=True, how_to_choose='medioid',
        feature_collapse_type='pca')
    combines['pred_voxlets_medioid'] = \
        save_render_assess({'name':'Voxlets medioid', 'grid':pred_voxlets_medioid, 'desc':'pred_voxlets_medioid'}, sc)
    pred_voxlets_medioid = None

    if only_prediction:
        return

    print '''DOING GREEDY ORACLES'''

    # true greedy oracle prediction - without the ground truth
    rec.initialise_output_grid(gt_grid=sc.gt_tsdf)
    true_greedy = rec.fill_in_output_grid_oma(
        render_type=[], add_ground_plane=True, feature_collapse_type='pca',
        oracle='true_greedy')

    for key, result in true_greedy.iteritems():
        dic = {'desc':'true_greedy_%04d' % key, 'name': 'True greedy %04d' % key, 'grid': result}
        combines['true_greedy_%04d' % key] = save_render_assess(dic, sc)
    true_greedy = None

    # hack to align the plotting
    # combines['blankblank'] = []

    # true greedy oracle prediction -with the ground truth
    rec.initialise_output_grid(gt_grid=sc.gt_tsdf)
    true_greedy_gt = rec.fill_in_output_grid_oma(
        render_type=[], add_ground_plane=True, feature_collapse_type='pca',
        oracle='true_greedy_gt')

    for key, result in true_greedy_gt.iteritems():
        dic = {'desc':'true_greedy_gt_%04d' % key, 'name': 'True greedy GT %04d' % key, 'grid': result}
        combines['true_greedy_gt_%04d' % key] = save_render_assess(dic, sc)
    true_greedy_gt = None

    if render_top_view:
        print "-> Rendering top view"
        rec.plot_voxlet_top_view(savepath=gen_renderpath % 'top_view')
        print gen_renderpath % 'top_view'

    '''DOING OTHER ORACLES'''

    if do_oracles:
        rec.initialise_output_grid(gt_grid=sc.gt_tsdf)
        full_oracle_voxlets = rec.fill_in_output_grid_oma(
            render_type=[],oracle='gt', add_ground_plane=True, feature_collapse_type='pca', use_binary=parameters.use_binary)
        combines['OR1'] = save_render_assess({'desc':'OR1', 'name':'Full oracle (OR1)', 'grid':full_oracle_voxlets}, sc)
        full_oracle_voxlets = None

        rec.initialise_output_grid(gt_grid=sc.gt_tsdf)
        oracle_voxlets = rec.fill_in_output_grid_oma(
            render_type=[],oracle='pca', add_ground_plane=True, feature_collapse_type='pca', use_binary=parameters.use_binary)
        combines['OR2'] = save_render_assess({'desc':'OR2', 'name':'Oracle using PCA (OR2)', 'grid':oracle_voxlets}, sc)
        oracle_voxlets = None

        rec.initialise_output_grid(gt_grid=sc.gt_tsdf)
        nn_oracle_voxlets = rec.fill_in_output_grid_oma(
            render_type=[], oracle='nn', add_ground_plane=True, feature_collapse_type='pca', use_binary=parameters.use_binary)
        combines['OR3'] = save_render_assess({'desc':'OR3', 'name':'Oracle using NN (OR3)', 'grid':nn_oracle_voxlets}, sc)
        nn_oracle_voxlets = None

    # if save_prediction_grids:
    #     print "-> Saving prediction grids"
    #     with open('/tmp/combines.pkl', 'w') as f:
    #         pickle.dump(combines, f, protocol=pickle.HIGHEST_PROTOCOL)

    # must save the input view to the save folder
    scipy.misc.imsave(gen_renderpath % 'input', sc.im.rgb)

    if combine_renders:
        print "-> Combining renders"
        su, sv = 3, 5

        fig = plt.figure(figsize=(25, 10), dpi=1000)
        plt.subplots(su, sv)
        plt.subplots_adjust(left=0, bottom=0, right=0.98, top=0.98, wspace=0.02, hspace=0.02)

        for count, (name, dic) in enumerate(combines.iteritems()):

            if count >= su*sv:
                raise Exception("Error! Final subplot is reserved for the ROC curve")
            if name == 'blankblank':
                plt.subplot(su, sv, count + 1)
                plt.imshow(np.zeros((3, 3)))
                plt.axis('off')
                continue

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
        plt.close()

        # Saving to the dropbox...
        if copy_to_dropbox:
            shutil.copy(all_savename, dropbox_path)

    if save_scores_to_yaml:
        print "-> Writing scores to YAML"
        print fpath + 'scores.yaml'
        results_dict = {}
        for name, dic in combines.iteritems():
            if name != 'input' and name != 'blankblank':
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
if parameters.multicore:
    import multiprocessing
    pool = multiprocessing.Pool(4)
    mapper = pool.map
else:
    mapper = map


if __name__ == '__main__':

    # temp = [s for s in paths.RenderedData.test_sequence() if s['name'] == 'd2p8ae7t0xi81q3y_SEQ']
    # print temp
    tic = time()
    mapper(process_sequence, paths.RenderedData.test_sequence()[10:48], chunksize=1)
    print "In total took %f s" % (time() - tic)


