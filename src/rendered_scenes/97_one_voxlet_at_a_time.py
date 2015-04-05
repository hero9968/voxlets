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
import copy

from common import paths
from common import parameters
from common import voxlets
from common import scene


# loading model
with open(paths.RenderedData.voxlet_model_oma_path, 'rb') as f:
    model = pickle.load(f)

# for count, tree in enumerate(model.forest.trees):
#     print count, len(tree.leaf_nodes())
#     lengths = np.array([len(leaf.exs_at_node) for leaf in tree.leaf_nodes()])
#     print np.sum(lengths<10), np.sum(np.logical_and(lengths>10, lengths<50)), np.sum(lengths>50)


test_types = ['oma']

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

    sc = scene.Scene(parameters.RenderedVoxelGrid.mu,
        model.voxlet_params)
    sc.load_sequence(sequence, frame_nos=0, segment_with_gt=True, save_grids=False)
    sc.santity_render(save_folder='/tmp/')

    test_type = 'oma'

    # fpath = paths.RenderedData.voxlet_prediction_folderpath % \
    #     (test_type, sequence['name'])
    # if not os.path.exists(fpath):
    #     os.makedirs(fpath)

    print "-> Reconstructing with oma forest"
    rec = voxlets.Reconstructer(
        reconstruction_type='kmeans_on_pca', combine_type='modal_vote')
    rec.set_model(model)
    rec.set_scene(sc)
    rec.sample_points(parameters.VoxletPrediction.number_samples,
                      parameters.VoxletPrediction.sampling_grid_size)
    rec.initialise_output_grid(gt_grid=sc.gt_tsdf)

    blank_rec = copy.deepcopy(rec)

    print "About to render scene..."
    sc.im_tsdf.render_view('/tmp/im_tsdf.png')

    # Here just complete one voxlet at a time...
    sampled_idxs = rec.sampled_idxs
    for count, idx in enumerate(sampled_idxs):

        rec = copy.deepcopy(blank_rec)

        rec.sampled_idxs = [idx]
        prediction = rec.fill_in_output_grid_oma( render_type=[], #['matplotlib'],
            render_savepath='/tmp/renders/', feature_collapse_type='pca')
        prediction_keeping_exisiting = rec.keeping_existing

        # creating a folder for this prediction...
        gen_renderpath = paths.RenderedData.voxlet_prediction_img_path % \
            (test_type, sequence['name'], ('voxlet_number_%04d/%s' % (count, '%s')))
        print gen_renderpath

        if not os.path.exists(os.path.dirname(gen_renderpath)):
            os.makedirs(os.path.dirname(gen_renderpath))

        # Rendering the prediction
        renderpath = gen_renderpath % '_diff'
        prediction_keeping_exisiting.render_view(renderpath)
        print "About to render..."
        #prediction.render_view(renderpath)
        voxlets.render_diff_view(prediction_keeping_exisiting, prediction, renderpath)

        # Rendering the extracted voxlet
        feature_voxlet_renderpath = gen_renderpath % '_feature'
        voxlets.render_single_voxlet(
            rec.cached_feature_voxlet.reshape(parameters.Voxlet.shape),
            feature_voxlet_renderpath)

        # Rendering the predicted voxlets
        pred_idxs = rec.model._cached_predictions[0]

        for pred_idx in pred_idxs:
            Y_pred_compressed = rec.model.training_Y[pred_idx]
            this_voxlet = rec.model.pca.inverse_transform(Y_pred_compressed)

            # compute the distance to the extracted feature voxlet...
            dims_to_use_for_distance = \
                rec.cached_feature_voxlet.flatten() == -np.array(parameters.RenderedVoxelGrid.mu).astype(rec.cached_feature_voxlet.dtype)
            print dims_to_use_for_distance.sum() / dims_to_use_for_distance.flatten().shape[0]
            distance = np.linalg.norm(
                rec.cached_feature_voxlet.flatten()[dims_to_use_for_distance] -
                this_voxlet.flatten()[dims_to_use_for_distance])

            prediction_voxlet_renderpath = gen_renderpath % (
                '_prediction_%04f_%04d' % (distance, pred_idx))
            voxlets.render_single_voxlet(
                this_voxlet.reshape(parameters.Voxlet.shape),
                prediction_voxlet_renderpath)

    # Hack to put in a floor
    prediction_keeping_exisiting.V[:, :, :4] = -1
    prediction.V[:, :, :4] = -1


    print "-> Saving"
    savepath = paths.RenderedData.voxlet_prediction_path % \
        (test_type, sequence['name'])
    prediction.save(savepath)
    savepath = paths.RenderedData.voxlet_prediction_path % \
        (test_type, sequence['name'] + '_keep_existing')
    prediction_keeping_exisiting.save(savepath)

    print "-> Rendering"
    renderpath = paths.RenderedData.voxlet_prediction_img_path % \
        (test_type, sequence['name'])
    renderpath = paths.RenderedData.voxlet_prediction_img_path % \
        (test_type, sequence['name'] + '_keep_existing')
    prediction_keeping_exisiting.render_view(renderpath)

    print "-> Done test type " + test_type

    print "Done sequence %s" % sequence['name']


# need to import these *after* the pool helper has been defined
if False: #parameters.multicore:
    import multiprocessing
    pool = multiprocessing.Pool(parameters.cores)
    mapper = pool.map
else:
    mapper = map


if __name__ == '__main__':

    tic = time()
    mapper(process_sequence, paths.RenderedData.test_sequence()[:5])
    print "In total took %f s" % (time() - tic)


