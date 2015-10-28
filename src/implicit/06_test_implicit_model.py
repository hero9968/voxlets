import sys
import os
import numpy as np
import yaml
import scipy.io
import sklearn.ensemble
import cPickle as pickle
import scipy.misc  # for saving images
from time import time
from sklearn.neighbors import NearestNeighbors

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.append(os.path.expanduser('~/projects/shape_sharing/src/'))

from common import voxel_data
from common import carving
from common import images
from common import scene
from features import line_casting

import system_setup

parameters = yaml.load(open('./implicit_params.yaml'))

if parameters['testing_data'] == 'oisin_house':
    import real_data_paths as paths
elif parameters['testing_data'] == 'synthetic':
    import synthetic_paths as paths
elif parameters['training_data'] == 'nyu_cad':
    import nyu_cad_paths as paths
else:
    raise Exception('Unknown training data')


render = True
save_training_pairs = False

def plot_slice(V):
    "Todo - move to the voxel class?"
    Aidx = V.shape[0]/2
    Bidx = V.shape[1]/2
    Cidx = V.shape[2]/2
    A = np.flipud(V[Aidx, :, :].T)
    B = np.flipud(V[:, Bidx, :].T)
    C = np.flipud(V[:, :, Cidx])
    bottom = np.concatenate((A, B), axis=1)
    tt = np.zeros((C.shape[0], B.shape[1])) * np.nan
    top = np.concatenate((C, tt), axis=1)
    plt.imshow(np.concatenate((top, bottom), axis=0))
    plt.colorbar()


def save_plot_slice(V, GT_V, imagesavepath, imtitle=""):
    "Create an overview image and save that to disk"

    plt.clf()
    plt.subplot(121)
    plot_slice(V)
    plt.title(imtitle)
    plt.subplot(122)
    plot_slice(GT_V)
    plt.title("Ground truth")

    "Saving figure"
    plt.savefig(imagesavepath, bbox_inches='tight')


def process_sequence(input_args):

    # unpack the input arguments
    rf, sequence, savename, oracle = input_args

    print "Loading sequence %s" % sequence['name']

    # this is where to save the results...
    results_foldername = \
        paths.implicit_predictions_dir % (savename, sequence['name'])
    if not os.path.exists(results_foldername):
        print "Creating %s" % results_foldername
        os.makedirs(results_foldername)

    # writing the yaml file to the test folder...
    with open(results_foldername + '/info.yaml', 'w') as f:
        yaml.dump(sequence, f)

    print "Processing " + sequence['name']
    sc = scene.Scene(parameters['mu'], None)
    sc.load_sequence(sequence, frame_nos=0, segment_with_gt=False, segment=False, save_grids=False)
    sc.im.mask = ~np.isnan(sc.im.depth)

    # getting the known full and empty voxels based on the depth image
    known_full_voxels = sc.im_visible
    known_empty_voxels = sc.im_tsdf.blank_copy()
    known_empty_voxels.V = sc.im_tsdf.V > 0

    # saving known empty and known full for my perusal
    scipy.io.savemat(
        results_foldername + 'known_voxels.mat',
        dict(known_full=known_full_voxels.V.astype(np.float64),
             known_empty=known_empty_voxels.V.astype(np.float64)),
        do_compression=True)

    # print "Computing the features"
    try:
        rays, Y, voxels_to_use = line_casting.feature_pairs_3d(
            known_empty_voxels,
            known_full_voxels,
            sc.gt_tsdf.V,
            in_frustrum=sc.get_visible_frustrum(),
            samples=-1, base_height=0)

        X = []
        if 'rays' in rf.parameters['features']:
            rays = line_casting.postprocess_features(
                rays, rf.parameters['postprocess'])
            X.append(rays)

        if 'cobweb' in rf.parameters['features']:
            cobweb = line_casting.cobweb_distance_features(
            sc, voxels_to_use, parameters['cobweb_offset'])
            cobweb[np.isnan(cobweb)] = parameters['cobweb_out_of_range']
            X.append(cobweb)

        # combining the features
        X = np.hstack(X)

    except:
        print "FAILURE! Skipping"
        return

    # print "Making prediction"
    if oracle != 'none':
        max_pairs = 200000
        idxs = np.random.choice(X.shape[0], max_pairs)
        X_subset = X[idxs, :]
        Y_subset = Y[idxs]

        if oracle == 'retrain_model':
            rf.fit(X_subset.astype(np.float32), Y_subset)
            Y_pred = rf.predict(X.astype(np.float32))

        elif oracle == 'nn':
            print "Retraining model, X is shape ", X_subset.shape
            nbrs = NearestNeighbors(n_neighbors=1, algorithm='kd_tree').fit(X_subset)

            print "Now fitting..."
            distances, indices = nbrs.kneighbors(X)
            Y_pred = Y_subset[indices[:, 0]]
    else:
        Y_pred = rf.predict(X.astype(np.float32))

    # now recreate the input image from the predictions
    # unknown_voxel_idxs = np.logical_and(
    #     known_empty_voxels.V.flatten() == 0,
    #     known_full_voxels.V.flatten() == 0)
    unknown_voxel_idxs_full = np.logical_and(
        known_empty_voxels.V == 0,
        known_full_voxels.V == 0)
    if save_training_pairs:
        training_pairs = dict(X=X, Y=Y,
            known_full=known_full_voxels.V.astype(np.float32),
            known_empty=known_empty_voxels.V.astype(np.float32),
            gt_vox=sc.gt_tsdf.V.astype(np.float32))
        scipy.io.savemat(
            results_foldername+'training_pairs.mat',
            training_pairs,
            do_compression=True)

    pred_grid = sc.im_tsdf.copy()

    pred_grid.set_indicated_voxels(voxels_to_use, Y_pred)

    # saving
    # print "Saving result to disk"
    pred_grid.save(results_foldername + 'prediction.pkl')
    scipy.io.savemat(
        results_foldername + 'prediction.mat',
        dict(gt=sc.gt_tsdf.V, pred=pred_grid.V, partial=sc.im_tsdf.V,
            known_full=known_full_voxels.V, known_empty=known_empty_voxels.V, dim=sc.im.rgb, rgbim=sc.im.rgb),
        do_compression=True)

    # print "Saving the input image"
    img_savepath = results_foldername + 'input_im.png'
    scipy.misc.imsave(img_savepath, sc.im.rgb)

    if render:
        print "Doing the rendering"
        pred_grid.render_view(results_foldername + 'prediction_render.png',
            xy_centre=True, ground_height=0.03, keep_obj=True)
        sc.im_tsdf.render_view(results_foldername + 'visible_render.png',
            xy_centre=True, keep_obj=True)
        sc.gt_tsdf.render_view(results_foldername + 'gt_render.png',
            xy_centre=True, keep_obj=True)

    # print "Evaluating"
    results = sc.evaluate_prediction(pred_grid.V)
    yaml.dump(results, open(results_foldername + 'eval.yaml', 'w'))

    img_savepath = results_foldername + 'prediction.png'
    save_plot_slice(pred_grid.V, sc.gt_tsdf.V, img_savepath, imtitle="")

    # print "Done "


if __name__ == '__main__':


    for model_params in parameters['models']:

        tic = time()

        print "Loading the model"
        loadpath = paths.implicit_model_dir % model_params['modelname'] + 'model.pkl'
        with open(loadpath, 'rb') as f:
            rf = pickle.load(f)

        savename = model_params['name']
        oracle = model_params['oracle']
        per_run_args = ((rf, seq, savename, oracle) for seq in paths.test_data)
        print len(rf.estimators_)
        # print list(per_run_args)

        if system_setup.multicore:
            import multiprocessing
            pool = multiprocessing.Pool(system_setup.testing_cores)
            pool.map(process_sequence, per_run_args)
            pool.close()
            pool.join()
        else:
            map(process_sequence, per_run_args)

        print "This model took %f s" % (time() - tic)
