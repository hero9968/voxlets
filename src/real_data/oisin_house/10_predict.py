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
import scipy.io
import yaml
import shutil
import collections

import real_data_paths as paths
import real_params as parameters

from common import voxlets
from common import scene

import sklearn.metrics

print "Loading model..."
cobweb = True
print paths.voxlet_model_oma_path

if cobweb:
    with open(paths.voxlet_model_oma_path.replace('.pkl', '_cobweb.pkl'), 'rb') as f:
        model_without_implicit = pickle.load(f)
else:
    with open(paths.voxlet_model_oma_path, 'rb') as f:
        model_without_implicit = pickle.load(f)

print model_without_implicit.voxlet_params['shape']
print model_without_implicit.voxlet_params['tall_voxlets']

print "Done loading model..."

render_gt = True
combining_renders = True
render_predictions = True
render_top_view = False
do_greedy_oracles = False
# save_prediction_grids = True
save_simple = True
save_scores_to_yaml = True
distance_experiments = True


copy_to_dropbox = False and paths.host_name == 'biryani'
dropbox_path = paths.new_dropbox_dir()
print dropbox_path


# this overrides all other parameters. Means we don't botther with orables etc
only_prediction = False

test_type = 'double_training_data_floating_voxlets'

def save_render_assess(dic, sc):
    '''
    does whatever is required in terms of saving results, rendering them etc
    '''

    # Path where any renders will be saved to
    gen_renderpath = paths.voxlet_prediction_img_path % \
        (test_type, sc.sequence['name'], '%s')

    if render_predictions:
        dic['grid'].render_view(gen_renderpath % dic['desc'], xy_centre=True,ground_height=0.03)

    if save_simple:
        D = dict(grid=dic['grid'].V, name=dic['desc'], vox_size=dic['grid'].vox_size)
        scipy.io.savemat(gen_renderpath.replace('png', 'mat') % dic['desc'], D)

    dic['results'] = sc.evaluate_prediction(dic['grid'].V)

    # finally removing the grid
    dic['grid'] = []

    return dic


def combine_renders(rec, combines, su, sv, gen_renderpath, savename):

    fig = plt.figure(figsize=(25, 10), dpi=1000)
    plt.subplots(su, sv)
    plt.subplots_adjust(left=0, bottom=0, right=0.98, top=0.98, wspace=0.02, hspace=0.02)

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

    plt.savefig(savename, dpi=400)
    plt.close()


print "MAIN LOOP"
# Note: if parallelising, should either do here (at top level) or at the
# bottom level, where the predictions are accumulated (although this might be)
# better off being GPU...)
def process_sequence(sequence):

    print "Processing ", sequence['name']
    sc = scene.Scene(parameters.mu,
        model_without_implicit.voxlet_params)
    sc.load_sequence(sequence, frame_nos=0, segment_with_gt=True,
        save_grids=False, load_implicit=False, voxel_normals='gt_tsdf')

    print "-> Creating folder"
    fpath = paths.voxlet_prediction_folderpath % \
        (test_type, sequence['name'])
    if not os.path.exists(fpath):
        os.makedirs(fpath)
    print "-> Doing prediction"
    # sc.santity_render(save_folder='/tmp/')

    combines = collections.OrderedDict()
    if render_gt:
        combines['input'] = {'name':'Input image'}
        combines['gt'] = save_render_assess(
            {'name':'Ground truth', 'grid':sc.gt_tsdf, 'desc':'gt'}, sc)
        combines['visible'] = save_render_assess(
            {'name':'Visible surfaces', 'grid':sc.im_tsdf, 'desc':'visible'}, sc)

    print "-> Sampling the points"
    rec = voxlets.Reconstructer(
        reconstruction_type='kmeans_on_pca', combine_type='modal_vote')
    rec.set_scene(sc)
    rec.sample_points(parameters.VoxletPrediction.number_samples,
                      parameters.VoxletPrediction.sampling_grid_size)

    # Path where any renders will be saved to
    gen_renderpath = paths.voxlet_prediction_img_path % \
        (test_type, sequence['name'], '%s')

    print "-> Saving the input rgb, depth and mask"
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

    print "-> Predicting with OMA forest"
    rec.initialise_output_grid(gt_grid=sc.gt_tsdf)
    rec.set_model(model_without_implicit)
    pred_voxlets = rec.fill_in_output_grid_oma(
        add_ground_plane=False, feature_collapse_type='pca', render_type=[],
        weight_empty_lower=None, cobweb=cobweb)
    # 'slice', 'tree_predictions'], render_savepath=fpath)

    pred_voxlets_exisiting = rec.keeping_existing
    pred_remove_excess = rec.remove_excess

    print "-> Doing the empty voxlet thing"
    rec.save_empty_voxel_counts(fpath)

    combines['pred_voxlets'] = save_render_assess({'name':'Voxlets', 'grid':pred_voxlets, 'desc':'pred_voxlets'}, sc)
    combines['pred_voxlets_exisiting'] = save_render_assess(
        {'name':'Voxlets existing', 'grid':pred_voxlets_exisiting, 'desc':'pred_voxlets_exisiting'}, sc)
    combines['pred_remove_excess'] = save_render_assess(
        {'name':'Voxlets removed excess', 'grid':pred_remove_excess, 'desc':'pred_remove_excess'}, sc)

    if render_top_view:
        print "-> Rendering top view"
        rec.plot_voxlet_top_view(savepath=gen_renderpath % 'top_view')
        print gen_renderpath % 'top_view'

    if only_prediction:
        return

    descs = ['narrow_band', 'mean', 'medioid']
    if distance_experiments:
        # here I'm doing an experiment to see which distance measure makes sense to use...
        names = ['Mean', 'Narrow Band', 'Medioid']
        for d_measure, name in zip(descs, names):
            rec.initialise_output_grid(gt_grid=sc.gt_tsdf)
            narrow_band = rec.fill_in_output_grid_oma(
                add_ground_plane=False, feature_collapse_type='pca', distance_measure=d_measure, cobweb=cobweb)
            combines[name] = save_render_assess(
                {'name':name, 'grid':narrow_band, 'desc':name}, sc)



    # must save the input view to the save folder
    scipy.misc.imsave(gen_renderpath % 'input', sc.im.rgb)

    if combining_renders:
        print "-> Combining renders"
        su, sv = 3, 3

        fname = 'all_' + rec.sc.sequence['name']
        all_savename = gen_renderpath.replace('png', 'pdf') % fname

        combine_renders(rec, combines, su, sv, gen_renderpath, all_savename)

        # Saving to the dropbox...
        if copy_to_dropbox:
            shutil.copy(all_savename, dropbox_path)

    if do_greedy_oracles:
        print "-> DOING GREEDY ORACLES"

        # true greedy oracle prediction - without the ground truth
        greedy_combines = collections.OrderedDict()
        greedy_combines['gt'] = save_render_assess(
            {'name':'Ground truth', 'grid':sc.gt_tsdf, 'desc':'gt'}, sc)
        greedy_combines['visible'] = save_render_assess(
            {'name':'Visible surfaces', 'grid':sc.im_tsdf, 'desc':'visible'}, sc)

        for d_measure in ['narrow_band', 'largest_of_free_zones', 'just_empty']:

            print "in outer loop" , d_measure
            rec.initialise_output_grid(gt_grid=sc.gt_tsdf)
            true_greedy = rec.fill_in_output_grid_oma(
                render_type=[], add_ground_plane=False, feature_collapse_type='pca',
                oracle='true_greedy', distance_measure=d_measure, cobweb=cobweb)

            for key, result in true_greedy.iteritems():
                print "In innder loop ", key
                dic = {'desc':'greedy_%s_%04d' % (d_measure, key),
                       'name': 'Greedy %s %04d' % (d_measure, key),
                       'grid': result}
                greedy_combines['greedy_%s_%04d' % (d_measure, key)] = \
                    save_render_assess(dic, sc)
            true_greedy = None

        # now combining the greedy renders...
        fname = 'all_greedy_' + rec.sc.sequence['name']
        all_savename = gen_renderpath.replace('png', 'pdf') % fname

        print "-> COMBINING GREEDY ORACLES"
        combine_renders(rec, greedy_combines, 4, 5, gen_renderpath, all_savename)

        # hack to align the plotting
        # combines['blankblank'] = []

        # # true greedy oracle prediction -with the ground truth
        # rec.initialise_output_grid(gt_grid=sc.gt_tsdf)
        # true_greedy_gt = rec.fill_in_output_grid_oma(
        #     render_type=[], add_ground_plane=True, feature_collapse_type='pca',
        #     oracle='true_greedy_gt')

        # for key, result in true_greedy_gt.iteritems():
        #     dic = {'desc':'true_greedy_gt_%04d' % key, 'name': 'True greedy GT %04d' % key, 'grid': result}
        #     combines['true_greedy_gt_%04d' % key] = save_render_assess(dic, sc)
        # true_greedy_gt = None

    if render_top_view:
        print "-> Rendering top view"
        rec.plot_voxlet_top_view(savepath=gen_renderpath % 'top_view')
        print gen_renderpath % 'top_view'

    if save_scores_to_yaml:
        print "-> Writing scores to YAML"
        results_dict = {}
        for name, dic in combines.iteritems():
            if name != 'input':
                test_key = name
                results_dict[test_key] = \
                    {'description': dic['name'],
                     'auc':         float(dic['results']['auc']),
                     'iou':         float(dic['results']['iou']),
                     'precision':   float(dic['results']['precision']),
                     'recall':      float(dic['results']['recall'])}

        with open(fpath + 'scores.yaml', 'w') as f:
            f.write(yaml.dump(results_dict, default_flow_style=False))


    print "Done sequence %s" % sequence['name']


# need to import these *after* the pool helper has been defined
if True:
    # parameters.multicore:
    import multiprocessing
    pool = multiprocessing.Pool(3)
    mapper = pool.map
else:
    mapper = map


if __name__ == '__main__':

    # print "Warning - just doing this one"
    # this_one = []
    # for t in paths.test_data:
    #     if t['name'] == 'saved_00196_[92]':
    #         this_one.append(t)
    # temp = [s for s in paths.RenderedData.test_sequence() if s['name'] == 'd2p8ae7t0xi81q3y_SEQ']
    # print temp
    tic = time()
    mapper(process_sequence, paths.test_data)
    # [:48], chunksize=1)
    print "In total took %f s" % (time() - tic)


