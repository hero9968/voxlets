import numpy as np
import scipy.io
import cPickle as pickle
import sys, os
sys.path.append(os.path.expanduser('~/projects/shape_sharing/src/'))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt 

from common import paths
from common import voxel_data
from common import images
from features import line_casting

# setting up paths here
rf_path = paths.base_path + '/implicit/bigbird/rf/rf_shallow_rotated.pkl'

print "Loading model"
rf = pickle.load(open(rf_path, 'rb'))

intrinsic_save_path = paths.base_path + '/implicit/bigbird/predictions_rotated/%s_%s.mat'
intrinsic_img_save_path = paths.base_path + '/implicit/bigbird/predictions_rotated/%s_%s.png'


def plot_slice(V):
    "Todo - move to the voxel class?"
    Aidx = V.shape[0]/2
    Bidx = V.shape[0]/2
    Cidx = V.shape[0]/2
    A = np.flipud(V[Aidx, :, :].T)
    B = np.flipud(V[:, Bidx, :].T)
    C = np.flipud(V[:, :, Cidx])
    bottom = np.concatenate((A, B), axis=1)
    tt = np.zeros((C.shape[0], B.shape[1])) * np.nan
    top = np.concatenate((C, tt), axis=1)
    plt.imshow( np.concatenate((top, bottom), axis=0))
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




for modelname in paths.test_names:
    
    print "Loading grid for " + modelname

    # loading ground truth voxel grid
    gt_grid = voxel_data.BigBirdVoxels()
    gt_grid.load_bigbird(modelname)

    expanded_gt = voxel_data.expanded_grid_accum(gt_grid)
    expanded_gt.fill_from_grid(gt_grid)

    gt_tsdf_V = expanded_gt.compute_tsdf(0.03)

    for this_view_idx in [0, 5, 10, 15, 20, 25, 30, 35, 40, 45]:


        test_view = paths.views[this_view_idx]        
        if os.path.exists(intrinsic_img_save_path % (modelname, test_view)):
            print "Skipping " + str(this_view_idx)
            continue

        test_im = images.CroppedRGBD()
        test_im.load_bigbird_from_mat(modelname, test_view)        

        # getting the known full and empty voxels based on the depth image
        known_full_voxels = voxel_data.get_known_full_grid(test_im, expanded_gt)
        known_empty_voxels = voxel_data.get_known_empty_grid(test_im, expanded_gt, 
                                                            known_full_voxels)[0]

        # extracting features
        X, Y = line_casting.feature_pairs_3d(known_empty_voxels, 
            known_full_voxels, gt_tsdf_V, samples=-1, base_height=15, autorotate=True)

        # making prediction
        Y_pred = rf.predict(X)

        # now recreate the input image from the predictions
        unknown_voxel_idxs = np.logical_and(known_empty_voxels.V.flatten() == 0,
                                                known_full_voxels.V.flatten() == 0)
        pred_grid = expanded_gt.blank_copy()
        pred_grid.V = pred_grid.V.astype(np.float64) + 0.03
        print np.sum(known_empty_voxels.V.flatten())
        print np.sum(known_empty_voxels.V.flatten())
        print unknown_voxel_idxs.shape
        print np.sum(unknown_voxel_idxs==1)
        print Y_pred.shape
        print Y_pred.dtype
        print pred_grid.V.shape
        pred_grid.set_indicated_voxels(unknown_voxel_idxs==1, Y_pred)

        # saving
        "Saving result to disk"
        savepath = intrinsic_save_path % (modelname, test_view)
        D = dict(prediction=pred_grid.V, gt=expanded_gt.V)
        scipy.io.savemat(savepath, D, do_compression=True)

        img_savepath = intrinsic_img_save_path % (modelname, test_view)
        save_plot_slice(pred_grid.V, gt_tsdf_V, img_savepath, imtitle="")

        print "Done " + test_view
        
        
        