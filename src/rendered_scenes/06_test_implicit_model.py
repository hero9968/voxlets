import sys
import os
import numpy as np
import yaml
import scipy.io
import sklearn.ensemble
import cPickle as pickle

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.append(os.path.expanduser('~/projects/shape_sharing/src/'))
sys.path.append(os.path.expanduser('~/projects/shape_sharing/src/intrinsic'))
from common import paths
from common import voxel_data
from common import carving
from common import images
from common import images
from features import line_casting

print "Loading the model"
rf_path = paths.implicit_models_folder + 'model.pkl'
rf = pickle.load(open(rf_path, 'rb'))

with open(paths.yaml_test_location, 'r') as f:
    test_sequences = yaml.load(f)


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


for sequence in test_sequences:

    print "Loading sequence %s" % sequence

    # this is where to save the results...
    results_foldername = \
        paths.test_sequences_save_location + sequence['name'] + '/'
    print "Creating %s" % results_foldername
    if not os.path.exists(results_foldername):
        os.makedirs(results_foldername)

    # this is where to
    input_data_path = paths.scenes_location + sequence['scene']
    vid = images.RGBDVideo()
    vid.load_from_yaml(input_data_path, 'poses.yaml')

    # load in the ground truth grid for this scene, and converting nans
    gt_vox = voxel_data.load_voxels(input_data_path + '/voxelgrid.pkl')
    gt_vox.V[np.isnan(gt_vox.V)] = -0.1#-0.03

    # do the tsdf fusion from these frames
    # note the slightly strange partial_tsdf copy and overright
    # need the copy for the R, origin etc. Don't give gt to the carver
    # to preserve train/test integrity!
    carver = carving.Fusion()
    carver.set_video(vid.subvid(sequence['frames']))
    partial_tsdf = gt_vox.blank_copy()
    carver.set_voxel_grid(partial_tsdf)
    partial_tsdf = carver.fuse(0.1)
#    print "Warning - test test"
#    partial_tsdf = gt_vox.copy()

    # save the grid
    partial_tsdf.save(results_foldername + 'input_fusion.pkl')

    # find the visible voxels ...? es ok
    vis = carving.VisibleVoxels()
    vis.set_voxel_grid(partial_tsdf)
    visible = vis.find_visible_voxels()

    # save the visible voxels
    visible.save(results_foldername + 'visible_voxels.pkl')

    # getting the known full and empty voxels based on the depth image
    known_full_voxels = visible
    known_empty_voxels = partial_tsdf.blank_copy()
    known_empty_voxels.V = partial_tsdf.V > 0

    # saving known empty and known full for my perusal
    scipy.io.savemat(
        results_foldername + 'known_voxels.mat',
        dict(
            known_full=known_full_voxels.V.astype(np.float64),
            known_empty=known_empty_voxels.V.astype(np.float64)))

    print "Computing the features"
    X, Y = line_casting.feature_pairs_3d(
        known_empty_voxels, known_full_voxels, gt_vox.V,
        samples=-1, base_height=0, autorotate=False)

    # saving these features to disk for my perusal
    scipy.io.savemat(results_foldername + 'features.mat', dict(X=X, Y=Y))

    print "Making prediction"
    Y_pred = rf.predict(X.astype())

    # now recreate the input image from the predictions
    unknown_voxel_idxs = np.logical_and(
        known_empty_voxels.V.flatten() == 0,
        known_full_voxels.V.flatten() == 0)
    unknown_voxel_idxs_full = np.logical_and(
        known_empty_voxels.V == 0,
        known_full_voxels.V == 0)
    scipy.io.savemat(
        results_foldername+'unknown_voxels.mat',
        dict(unknown=unknown_voxel_idxs_full.astype(np.float64)))
    #pred_grid = gt_vox.blank_copy()
    pred_grid = partial_tsdf.copy() #pred_grid.V.astype(np.float64) + 0.03

    pred_grid.set_indicated_voxels(unknown_voxel_idxs == 1, Y_pred)

    # saving
    "Saving result to disk"
    pred_grid.save(results_foldername + 'prediction.pkl')
    scipy.io.savemat(
        results_foldername + 'prediction.mat',
        dict(gt=gt_vox.V, pred=pred_grid.V, partial=partial_tsdf.V),
        do_compression=True)

    img_savepath = results_foldername + 'prediction.png'
    save_plot_slice(pred_grid.V, gt_vox.V, img_savepath, imtitle="")

    print "Done "
