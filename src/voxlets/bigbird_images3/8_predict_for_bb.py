'''
make predictions for all of bigbird dataset
using my algorhtm
saves each prediction to disk
'''

import numpy as np
import matplotlib.pyplot as plt 
import cPickle as pickle
import sys, os
sys.path.append(os.path.expanduser("~/projects/shape_sharing/src/"))
import copy
import sklearn.metrics

from common import paths
from common import voxel_data
from common import mesh
from common import images
from common import features
import reconstructer

print "Setting parameters"
max_points = 100
number_samples = 2000
combine_type = 'medioid'
reconstruction_type='kmeans_on_pca'

print "Loading clusters and forest"
forest_pca = pickle.load(open(paths.voxlet_model_pca_path, 'rb'))
km_pca = pickle.load(open(paths.voxlet_pca_dict_path, 'rb'))
pca = pickle.load(open(paths.voxlet_pca_path, 'rb'))

def pool_helper(gt_grid, test_im):

    print "Inside the helper"
    rec = reconstructer.Reconstructer(reconstruction_type='kmeans_on_pca', combine_type='medioid')
    rec.set_forest(forest_pca)
    rec.set_pca_comp(pca)
    rec.set_km_dict(km_pca)
    rec.set_test_im(test_im)
    rec.sample_points(number_samples)
    rec.initialise_output_grid(method='from_grid', gt_grid=gt_grid)
    accum1 = rec.fill_in_output_grid(max_points=max_points)
    
    rec = reconstructer.Reconstructer(reconstruction_type='kmeans_on_pca', combine_type='modal_vote')
    rec.set_forest(forest_pca)
    rec.set_pca_comp(pca)
    rec.set_km_dict(km_pca)
    rec.set_test_im(test_im)
    rec.sample_points(number_samples)
    rec.initialise_output_grid(method='from_grid', gt_grid=gt_grid)
    accum2 = rec.fill_in_output_grid(max_points=max_points)

    return accum1.compute_average(nan_value=0.03), accum2.compute_average(nan_value=0.03), accum1

"MAIN LOOP"
for modelname in paths.test_names:

    "Loading in test data"
    vgrid = voxel_data.BigBirdVoxels()
    vgrid.load_bigbird(modelname)

    for this_view_idx in [0, 5, 10, 15, 20, 25, 30, 35, 40, 45]:

        test_view = paths.views[this_view_idx]

        test_im = images.CroppedRGBD()
        test_im.load_bigbird_from_mat(modelname, test_view)

        result1, result2, accum1 = pool_helper(vgrid, test_im)

        "Saving result to disk"
        savepath = paths.voxlet_prediction_path % (modelname, test_view)
        D = dict(medioid=result1, modal=result2)
        f = open(savepath, 'wb')
        pickle.dump(D, f)
        f.close()

        "Create an overview image and save that to disk also"
        def plot_slice(V):
            A = np.flipud(V[15, :, :].T)
            B = np.flipud(V[:, 15, :].T)
            C = np.flipud(V[:, :, 30])
            bottom = np.concatenate((A, B), axis=1)
            tt = np.zeros((C.shape[0], B.shape[1])) * np.nan
            top = np.concatenate((C, tt), axis=1)
            plt.imshow( np.concatenate((top, bottom), axis=0))

        gt_out = copy.deepcopy(accum1)
        gt_out.V *= 0
        gt_out.fill_from_grid(vgrid)
        gt = gt_out.compute_tsdf(0.03)

        "Filling the figure"
        plt.rcParams['figure.figsize'] = (15.0, 20.0)
        plt.clf()
        plt.subplot(131)
        plot_slice(result1)
        plt.title('Medioid')
        plt.subplot(132)
        plot_slice(result2)
        plt.title('Modal')
        plt.subplot(133)
        plot_slice(gt)
        plt.title('Ground truth')

        "Computing the scores"
        # compute some kind of result... true positive? false negatuve>
        gt += 0.03
        gt /= 0.06
        gt = gt.astype(int)
        result1 += 0.03
        result1 /= 0.06
        result2 += 0.03
        result2 /= 0.06
        auc1 = sklearn.metrics.roc_auc_score(gt.flatten(), result1.flatten())
        auc2 = sklearn.metrics.roc_auc_score(gt.flatten(), result2.flatten())

        "Saving figure"
        imagesavepath = paths.voxlet_prediction_image_path % (modelname, test_view, auc1, auc2)
        plt.savefig(imagesavepath, bbox_inches='tight')

    print "DOne model " + modelname


