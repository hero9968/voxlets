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
import sys, os
sys.path.append(os.path.expanduser("~/projects/shape_sharing/src/"))
import copy
import sklearn.metrics
import scipy.io

from common import paths
from common import voxel_data
from common import mesh
from common import images
from common import features
import reconstructer
import baselines

print "Setting parameters"
max_points = 200
number_samples = 2000
multiproc = not paths.small_sample

test_types = ['bb']#['oma', 'modal', 'medioid', 'bb']#, 'bpc', 'no_spider']


if 'modal' in test_types or 'medioid' in test_types:

    print "Loading clusters and forest"
    forest_pca = pickle.load(open(paths.voxlet_model_pca_path, 'rb'))
    km_pca = pickle.load(open(paths.voxlet_pca_dict_path, 'rb'))
    pca = pickle.load(open(paths.voxlet_pca_path, 'rb'))

if 'oma' in test_types:

    print "Loading OMA forest"
    pca = pickle.load(open(paths.voxlet_pca_path, 'rb'))
    oma_forest = pickle.load(open(paths.voxlet_model_oma_path, 'rb'))
    #pass

if 'no_spider' in test_types:
    sa
    pass


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



def main_pool_helper(this_view_idx, modelname,  gt_grid, test_type):
    '''
    loads an image and then does the prediction for it 
    '''

    test_view = paths.views[this_view_idx]
    test_im = images.CroppedRGBD()
    test_im.load_bigbird_from_mat(modelname, test_view)


    if test_type == 'medioid':

        rec = reconstructer.Reconstructer(reconstruction_type='kmeans_on_pca', combine_type='medioid')
        rec.set_forest(forest_pca)
        rec.set_pca_comp(pca)
        rec.set_km_dict(km_pca)
        rec.set_test_im(test_im)
        rec.sample_points(number_samples)
        rec.initialise_output_grid(method='from_grid', gt_grid=gt_grid)
        accum = rec.fill_in_output_grid(max_points=max_points)
        prediction = accum.compute_average(nan_value=0.03)

    elif test_type == 'modal':

        rec = reconstructer.Reconstructer(reconstruction_type='kmeans_on_pca', combine_type='modal_vote')
        rec.set_forest(forest_pca)
        rec.set_pca_comp(pca)
        rec.set_km_dict(km_pca)
        rec.set_test_im(test_im)
        rec.sample_points(number_samples)
        rec.initialise_output_grid(method='from_grid', gt_grid=gt_grid)
        accum = rec.fill_in_output_grid(max_points=max_points)
        prediction = accum.compute_average(nan_value=0.03)

    elif test_type == 'bb':

        accum = voxel_data.expanded_grid_accum(gt_grid)
        accum.fill_from_grid(gt_grid)

        # Min area bounding box
        bb = baselines.BbBaseline()
        bb.set_gt_vgrid(accum)
        prediction = bb.compute_min_bb().V

    elif test_type == 'oma':

        print "Reconstructing with oma forest"
        rec = reconstructer.Reconstructer(reconstruction_type='kmeans_on_pca', combine_type='modal_vote')
        rec.set_forest(oma_forest)
        rec.set_pca_comp(pca)
        rec.set_test_im(test_im)
        rec.sample_points(number_samples)
        rec.initialise_output_grid(method='from_grid', gt_grid=gt_grid)
        accum = rec.fill_in_output_grid_oma(max_points=max_points)
        prediction = accum.compute_average(nan_value=0.03)

    elif test_type == 'oma_cobweb':

        print "Reconstructing with oma forest"
        rec = reconstructer.Reconstructer(reconstruction_type='kmeans_on_pca', combine_type='modal_vote')
        rec.set_forest(oma_forest)
        rec.set_pca_comp(pca)
        rec.set_test_im(test_im)
        rec.sample_points(number_samples)
        rec.initialise_output_grid(method='from_grid', gt_grid=gt_grid)
        accum = rec.fill_in_output_grid_oma(max_points=max_points, special='cobweb')
        prediction = accum.compute_average(nan_value=0.03)

    elif test_type == 'bpc':
        pass

    else:
        error("Unknown test type")

    "The ground truth is recreated so as to have the same padding that the test results have"
    gt_out = copy.deepcopy(accum)
    gt_out.V *= 0
    gt_out.fill_from_grid(gt_grid)
    gt = gt_out.compute_tsdf(0.03)

    "Saving result to disk"
    savepath = paths.voxlet_prediction_path % (test_type, modelname, test_view)
    D = dict(prediction=prediction, gt=gt)
    scipy.io.savemat(savepath, D, do_compression=True)

    "Computing the auc score"
    gt_occ = ((gt + 0.03) / 0.06).astype(int)
    prediction_occ = (prediction + 0.03) / 0.06
    auc = sklearn.metrics.roc_auc_score(gt_occ.flatten(), prediction_occ.flatten())

    "Filling the figure"
    imagesavepath = paths.voxlet_prediction_image_path % (test_type, modelname, test_view)
    save_plot_slice(prediction, gt, imagesavepath, imtitle=str(auc))
 
    print "Done view " + str(this_view_idx)
    

import multiprocessing
import functools

if multiproc:
    if paths.small_sample:
        pool = multiprocessing.Pool(4)
    else:
        pool = multiprocessing.Pool(6)


print "Checking results folders exist, creating if not"
for test_type in test_types:
    folder_save_path = paths.voxlet_prediction_folder_path % test_type
    if not os.path.exists(folder_save_path):
        os.makedirs(folder_save_path)


print "MAIN LOOP"
for modelname in paths.test_names:

    "Loading in test data"
    gt_grid = voxel_data.BigBirdVoxels()
    gt_grid.load_bigbird(modelname)

    poss_views = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45]

    for test_type in test_types:
        if multiproc:
            print "multiprocessing"
            part_func = functools.partial(main_pool_helper, modelname=modelname, gt_grid=gt_grid, test_type=test_type)
            pool.map(part_func, poss_views)
        else:
            print "not multiprocessing"
            for v in poss_views:
                main_pool_helper(v, modelname=modelname, gt_grid=gt_grid, test_type=test_type)
            print "Done test type " + test_type

    print "Done model " + modelname


