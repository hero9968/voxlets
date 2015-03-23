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

from common import paths
from common import parameters
from common import voxlets
from common import scene

import sklearn.metrics

# loading model
with open(paths.RenderedData.voxlets_path + '/models_implicit/oma.pkl', 'rb') as f:
    model_with_implicit = pickle.load(f)

with open(paths.RenderedData.voxlets_path + '/models/oma.pkl', 'rb') as f:
    model_without_implicit = pickle.load(f)



# for count, tree in enumerate(model.forest.trees):
#     print count, len(tree.leaf_nodes())
#     lengths = np.array([len(leaf.exs_at_node) for leaf in tree.leaf_nodes()])
#     print np.sum(lengths<10), np.sum(np.logical_and(lengths>10, lengths<50)), np.sum(lengths>50)


test_types = ['oma_implicit']

# print "Checking results folders exist, creating if not"
# for test_type in test_types + ['partial_tsdf', 'visible_voxels']:
#     print test_type
#     folder_save_path = \
#         paths.RenderedData.voxlet_prediction_path % (test_type, '_')
#     folder_save_path = os.path.dirname(folder_save_path)
#     print folder_save_path
#     if not os.path.exists(folder_save_path):
#         os.makedirs(folder_save_path)


print "MAIN LOOP"
# Note: if parallelising, should either do here (at top level) or at the
# bottom level, where the predictions are accumulated (although this might be)
# better off being GPU...)
def process_sequence(sequence):

    sc = scene.Scene()
    sc.load_sequence(sequence, frame_nos=0, segment_with_gt=True, save_grids=False, load_implicit=True)
    sc.santity_render(save_folder='/tmp/')

    test_type = 'oma_implicit'

    print "-> Reconstructing with oma forest"
    rec = voxlets.Reconstructer(
        reconstruction_type='kmeans_on_pca', combine_type='modal_vote')
    rec.set_scene(sc)
    rec.sample_points(parameters.VoxletPrediction.number_samples)

    # rec.initialise_output_grid(gt_grid=sc.gt_tsdf)
    # rec.set_model(model_with_implicit)
    # pred_voxlets_implicit = rec.fill_in_output_grid_oma( render_type=[], #['matplotlib'],
    #     render_savepath='/tmp/renders/', use_implicit=True)
    # pred_voxlets_implicit_exisiting = rec.keeping_existing

    rec.initialise_output_grid(gt_grid=sc.gt_tsdf)
    rec.set_model(model_without_implicit)
    pred_voxlets = rec.fill_in_output_grid_oma( render_type=[], #['matplotlib'],
        render_savepath='/tmp/renders/', use_implicit=False)
    pred_voxlets_exisiting = rec.keeping_existing

    rec.initialise_output_grid(gt_grid=sc.gt_tsdf)
    oracle_voxlets = rec.fill_in_output_grid_oma( render_type=[], #['matplotlib'],
        render_savepath='/tmp/renders/', use_implicit=False, oracle='pca')
    oracle_voxlets_existing = rec.keeping_existing

    rec.initialise_output_grid(gt_grid=sc.gt_tsdf)
    nn_oracle_voxlets = rec.fill_in_output_grid_oma( render_type=[],
        render_savepath='/tmp/renders/', use_implicit=False, oracle='nn')
    nn_oracle_voxlets_existing = rec.keeping_existing

    # Hack to put in a floor
    # pred_voxlets_implicit_exisiting.V[:, :, :4] = -parameters.RenderedVoxelGrid.mu
    pred_voxlets_exisiting.V[:, :, :4] = -parameters.RenderedVoxelGrid.mu
    pred_voxlets.V[:, :, :4] = -parameters.RenderedVoxelGrid.mu
    # pred_voxlets_implicit.V[:, :, :4] = -parameters.RenderedVoxelGrid.mu
    oracle_voxlets.V[:, :, :4] = -parameters.RenderedVoxelGrid.mu
    nn_oracle_voxlets.V[:, :, :4] = -parameters.RenderedVoxelGrid.mu

    print "-> Creating folder"
    fpath = paths.RenderedData.voxlet_prediction_folderpath % \
        (test_type, sequence['name'])
    if not os.path.exists(fpath):
        os.makedirs(fpath)

    print "-> Saving"
    # savepath = paths.RenderedData.voxlet_prediction_path % \
    #     (test_type, sequence['name'])
    # prediction.save(savepath)
    # savepath = paths.RenderedData.voxlet_prediction_path % \
    #     (test_type, sequence['name'], 'keep_existing')
    # prediction_keeping_exisiting.save(savepath)

    print "-> Rendering"
    gen_renderpath = paths.RenderedData.voxlet_prediction_img_path % \
        (test_type, sequence['name'], '%s')

    
    # pred_voxlets_implicit_exisiting.render_view(gen_renderpath % 'pred_voxlets_implicit_exisiting')
    # pred_voxlets_implicit.render_view(gen_renderpath % 'pred_voxlets_implicit')
    oracle_voxlets.render_view(gen_renderpath % 'oracle_voxlets')
    nn_oracle_voxlets.render_view(gen_renderpath % 'nn_oracle_voxlets')
    pred_voxlets_exisiting.render_view(gen_renderpath % 'pred_voxlets_exisiting')
    pred_voxlets.render_view(gen_renderpath % 'pred_voxlets')
    sc.implicit_tsdf.render_view(gen_renderpath % 'implicit')
    sc.im_tsdf.render_view(gen_renderpath % 'visible')
    sc.gt_tsdf.render_view(gen_renderpath % 'gt')
    scipy.misc.imsave(gen_renderpath % 'input', sc.im.rgb)

    combines = [
        ['Ground truth', 'gt'],
        ['Input image', 'input'],
        ['Visible surfaces', 'visible', sc.im_tsdf],
        ['Oracle using PCA', 'oracle_voxlets', oracle_voxlets],
        ['Oracle using NN', 'nn_oracle_voxlets', nn_oracle_voxlets],
        ['Implicit prediction', 'implicit', sc.implicit_tsdf],
        ['Voxlets', 'pred_voxlets', pred_voxlets],
        ['Voxlets + visible', 'pred_voxlets_exisiting', pred_voxlets_exisiting]]
        # ['Voxlets + visible, using implicit', 'pred_voxlets_implicit_exisiting', pred_voxlets_implicit_exisiting]]
        # ['Voxlets using implicit', 'pred_voxlets_implicit', pred_voxlets_implicit],

    with open('/tmp/combines.pkl', 'w') as f:
        pickle.dump(combines, f, protocol=pickle.HIGHEST_PROTOCOL)

    "Compute the score for each prediction"
    voxels_to_evaluate = np.logical_or(sc.im_tsdf.V < 0, np.isnan(sc.im_tsdf.V))
    gt = sc.gt_tsdf.V[voxels_to_evaluate] > 0
    gt[np.isnan(gt)] = -parameters.RenderedVoxelGrid.mu
    print gt.sum()

    print "Voxels to evaluate has shape  and sum: ", voxels_to_evaluate.shape, voxels_to_evaluate.sum()
    for c in combines[3:]:
        voxel_predictions = c[2].V[voxels_to_evaluate]
        voxel_predictions[np.isnan(voxel_predictions)] = \
            +parameters.RenderedVoxelGrid.mu

        print "Sum is " , c[0],  voxel_predictions.sum()

        score = sklearn.metrics.roc_auc_score(gt, voxel_predictions)
        fpr, tpr, _ = sklearn.metrics.roc_curve(gt, voxel_predictions)
        c.append([score, fpr, tpr])

    print [c[3] for c in combines[3:]]

    su, sv = 3, 3

    fig = plt.figure(figsize=(25, 10), dpi=1000)
    plt.subplots(su, sv)
    plt.subplots_adjust(left=0, bottom=0, right=0.95, top=0.95, wspace=0.2, hspace=0.2)

    for count, c in enumerate(combines):

        if count >= su*sv:
            raise Exception("Error! Final subplot is reserved for the ROC curve")

        plt.subplot(su, sv, count + 1)
        plt.imshow(scipy.misc.imread(gen_renderpath % c[1]))
        plt.axis('off')
        plt.title(c[0])

        " Add to the roc plot, which is in the final subplot"
        if count >= 3:
            "Add the AUC"
            plt.text(0, 50, "AUC = %0.3f" % c[3][0], fontsize=12, color='white')
            
            plt.subplot(su, sv, su*sv)
            plt.plot(c[3][1], c[3][2], label=c[0])
            plt.hold(1)
            plt.legend(prop={'size':6}, loc='lower right')

    fname = 'all_' + sequence['name']
    plt.savefig(gen_renderpath.replace('png', 'pdf') % fname, dpi=400)

    print "-> Done test type " + test_type

    print "Done sequence %s" % sequence['name']
    quit()

# need to import these *after* the pool helper has been defined
if False:  # parameters.multicore:
    import multiprocessing
    pool = multiprocessing.Pool(parameters.cores)
    mapper = pool.map
else:
    mapper = map


if __name__ == '__main__':

    temp = [s for s in paths.RenderedData.test_sequence() if s['name'] == 'd2p8ae7t0xi81q3y_SEQ']
    print temp
    tic = time()
    mapper(process_sequence, paths.RenderedData.test_sequence())
    print "In total took %f s" % (time() - tic)


