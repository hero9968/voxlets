import sys
import os
import numpy as np
import yaml
import scipy.io
import sklearn.ensemble
import cPickle as pickle
import scipy.misc  # for saving images
from time import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.append(os.path.expanduser('~/projects/shape_sharing/src/'))
sys.path.append(os.path.expanduser('~/projects/shape_sharing/src/intrinsic'))
from common import paths
from common import voxel_data
from common import carving
from common import images
from common import parameters
from common import scene
from features import line_casting

print "Loading the model"
with open(paths.RenderedData.implicit_models_dir + 'model.pkl', 'rb') as f:
    rf = pickle.load(f)

render = True


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


def process_sequence(sequence):

    print "Loading sequence %s" % sequence

    # this is where to save the results...
    results_foldername = \
        paths.RenderedData.implicit_prediction_dir % sequence['name']
    print "Creating %s" % results_foldername
    if not os.path.exists(results_foldername):
        os.makedirs(results_foldername)

    # writing the yaml file to the test folder...
    with open(results_foldername + '/info.yaml', 'w') as f:
        yaml.dump(sequence, f)

    print "Processing " + sequence['name']
    sc = scene.Scene()
    sc.load_sequence(sequence, frame_nos=0, segment_with_gt=False, segment=False, save_grids=False)

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

    print "Computing the features"
    X, Y = line_casting.feature_pairs_3d(
        known_empty_voxels, known_full_voxels, sc.gt_tsdf.V,
        samples=-1, base_height=0, autorotate=False)

    # saving these features to disk for my perusal
    scipy.io.savemat(results_foldername + 'features.mat', dict(X=X, Y=Y), do_compression=True)

    print "Making prediction"
    Y_pred = rf.predict(X.astype(np.float32))

    # now recreate the input image from the predictions
    unknown_voxel_idxs = np.logical_and(
        known_empty_voxels.V.flatten() == 0,
        known_full_voxels.V.flatten() == 0)
    unknown_voxel_idxs_full = np.logical_and(
        known_empty_voxels.V == 0,
        known_full_voxels.V == 0)
    training_pairs = dict(X=X, Y=Y,
        known_full=known_full_voxels.V.astype(np.float32),
        known_empty=known_empty_voxels.V.astype(np.float32),
        gt_vox=sc.gt_tsdf.V.astype(np.float32))
    scipy.io.savemat(
        results_foldername+'training_pairs.mat',
        training_pairs,
        do_compression=True)

    pred_grid = sc.im_tsdf.copy()

    pred_grid.set_indicated_voxels(unknown_voxel_idxs == 1, Y_pred)

    # saving
    "Saving result to disk"
    pred_grid.save(results_foldername + 'prediction.pkl')
    scipy.io.savemat(
        results_foldername + 'prediction.mat',
        dict(gt=sc.gt_tsdf.V, pred=pred_grid.V, partial=sc.im_tsdf.V,
            known_full=known_full_voxels.V, known_empty=known_empty_voxels.V, dim=sc.im.rgb, rgbim=sc.im.rgb),
        do_compression=True)

    "Saving the input image"
    img_savepath = results_foldername + 'input_im.png'
    scipy.misc.imsave(img_savepath, sc.im.rgb)

    if render:
        pred_grid.render_view(results_foldername + 'prediction_render.png')
        sc.im_tsdf.render_view(results_foldername + 'visible_render.png')
        sc.gt_tsdf.render_view(results_foldername + 'gt_render.png')

    img_savepath = results_foldername + 'prediction.png'
    save_plot_slice(pred_grid.V, sc.gt_tsdf.V, img_savepath, imtitle="")

    print "Done "


# need to import these *after* the pool helper has been defined
if parameters.multicore:
    import multiprocessing
    import functools
    pool = multiprocessing.Pool(parameters.cores)
    mapper = pool.map
else:
    mapper = map


if __name__ == '__main__':

    tic = time()
    print "DANGER - doing on train sequence"
    mapper(process_sequence, paths.RenderedData.test_sequence())
    print "In total took %f s" % (time() - tic)
