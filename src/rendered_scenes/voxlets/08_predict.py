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

from common import paths
from common import parameters
from common import voxel_data
from common import images
from common import features
from common import voxlets


# loading model
with open(paths.RenderedData.voxlet_model_oma_path, 'rb') as f:
    model = pickle.load(f)

test_types = ['oma']

print "Checking results folders exist, creating if not"
for test_type in test_types:
    folder_save_path = \
        paths.RenderedData.voxlet_prediction_path % (test_type, '_')
    folder_save_path = os.path.dirname(folder_save_path)
    print folder_save_path
    if not os.path.exists(folder_save_path):
        os.makedirs(folder_save_path)

print "MAIN LOOP"
# Note: if parallelising, should either do here (at top level) or at the
# bottom level, where the predictions are accumulated (although this might be)
# better off being GPU...)
for count, sequence in enumerate(paths.RenderedData.test_sequence()):

    # load in the ground truth grid for this scene, and converting nans
    vox_location = paths.RenderedData.ground_truth_voxels(sequence['scene'])
    gt_vox = voxel_data.load_voxels(vox_location)
    gt_vox.V[np.isnan(gt_vox.V)] = -parameters.RenderedVoxelGrid.mu
    gt_vox.set_origin(gt_vox.origin)

    # loading in the image
    frame_data = paths.RenderedData.load_scene_data(
        sequence['scene'], sequence['frames'][0])
    im = images.RGBDImage.load_from_dict(
        paths.RenderedData.scene_dir(sequence['scene']),
        frame_data)

    # computing normals...
    norm_engine = features.Normals()
    im.normals = norm_engine.compute_normals(im)

    for test_type in test_types:

        print "Reconstructing with oma forest"
        rec = voxlets.Reconstructer(
            reconstruction_type='kmeans_on_pca', combine_type='modal_vote')
        rec.set_model(model)
        rec.set_test_im(im)
        rec.sample_points(parameters.VoxletPrediction.number_samples)
        rec.initialise_output_grid(gt_grid=gt_vox)
        accum = rec.fill_in_output_grid_oma()
        prediction = accum.compute_average(
            nan_value=parameters.RenderedVoxelGrid.mu)

        print "Saving"
        savepath = paths.RenderedData.voxlet_prediction_path % \
            (test_type, sequence['name'])
        prediction.save(savepath)

        print "Done test type " + test_type

    print "Done sequence %s" % sequence['name']

    # "Saving result to disk"
    # savepath =

    # savepath = paths.voxlet_prediction_path % \
    # (test_type, modelname, test_view)
    # D = dict(prediction=prediction.V, gt=gt)
    # scipy.io.savemat(savepath, D, do_compression=True)

    # "Now also save to a pickle file so have the original data..."
    # savepathpickle = paths.voxlet_prediction_path_pkl % \
    #     (test_type, modelname, test_view)
    # pickle.dump(prediction, open(savepathpickle, 'wb'))

    # "Computing the auc score"
    # gt_occ = ((gt + 0.03) / 0.06).astype(int)
    # prediction_occ = (prediction.V + 0.03) / 0.06
    # auc = sklearn.metrics.roc_auc_score(
    #     gt_occ.flatten(), prediction_occ.flatten())

    # "Filling the figure"
    # imagesavepath = paths.voxlet_prediction_image_path % \
    #     (test_type, modelname, test_view)
    # save_plot_slice(prediction.V, gt, imagesavepath, imtitle=str(auc))

    # # need to do this here after the pool helper has been defined...
    # import multiprocessing
    # import functools

    # if multiproc:
    #     pool = multiprocessing.Pool(parameters.cores)
