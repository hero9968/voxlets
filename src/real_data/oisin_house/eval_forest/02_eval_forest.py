'''
code to test a model and report the score
'''
import numpy as np
import cPickle as pickle
import sys
import os
sys.path.append(os.path.expanduser("~/projects/shape_sharing/src/"))
sys.path.append(os.path.expanduser("~/projects/shape_sharing/src/real_data/"))
from time import time
import yaml
import functools
import collections
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import pairwise_distances
from oisin_house import real_data_paths as paths
from common import voxlets, scene, rendering
from oisin_house import system_setup

from copy import copy

parameters_path = '../training_params.yaml'
parameters = yaml.load(open(parameters_path))


def load_eval_data(voxlet_name, feature_type):
    '''
    Loading in all the data...
    '''
    features = []
    samples = []
    pca_representation = []

    for count, sequence in enumerate(paths.test_data):

        loadfolder = paths.evaluation_data_path % voxlet_name
        loadpath = loadfolder + sequence['name'] + '.pkl'
        D = pickle.load(open(loadpath, 'r'))

        features.append(D[feature_type])
        pca_representation.append(D['shoeboxes'])

    Y = np.vstack(pca_representation)
    X = np.concatenate(features, axis=0)

    print "\tVoxlets is\t", Y.shape
    print "\tFeatures is\t", X.shape

    return X, Y

def evaluate(gt_in, prediction_in, prediction_mask):

    to_use = prediction_mask < 0.5
    gt = gt_in[to_use] < 0
    prediction = prediction_in[to_use] < 0

    union = np.logical_or(gt, prediction)
    intersection = np.logical_and(gt, prediction)

    tp = float(np.logical_and(gt, prediction).sum())
    tn = float(np.logical_and(~gt, ~prediction).sum())
    fp = float(np.logical_and(~gt, prediction).sum())
    fn = float(np.logical_and(gt, ~prediction).sum())

    results = {}
    results['iou'] = float(intersection.sum()) / float(union.sum())
    results['precision'] = tp / (tp + fp + 0.0000001)
    results['recall'] = tp / (tp + fn + 0.0000001)
    return results


if __name__ == '__main__':

    results = collections.OrderedDict()
    scores = ['iou', 'precision', 'recall']

    # loop over tall v short voxlets
    for model_name in ['short_cobweb', 'tall_cobweb']:

        # loop over feature type
        for feature_type in ['mahalanobis', 'gt_oracle_full', 'gt_oracle', 'samples', 'cobweb']:

            print "--> Model name %s, feature type %s" % (model_name, feature_type)

            print "--> Loading models..."
            loadpath = paths.voxlet_model_path % model_name
            model = pickle.load(open(loadpath.replace('.pkl', '_full.pkl')))

            if 'gt_oracle' in feature_type or 'mahalanobis' in feature_type:
                print "--> Downsampling the model data for speed and testing!"
                to_use = np.random.choice(model.training_Y.shape[0], 1000)
                model.training_Y = model.training_Y[to_use]
                model.training_masks = model.training_masks[to_use]

            print "--> Loading data associated with this model"
            X, Y_before_pca = load_eval_data(vox_type, model_name)

            X[np.isnan(X)] = \
                float(parameters[model_name + '_out_of_range_feature'])

            print "--> Downsampling"
            max_to_use = 100
            X = X[:max_to_use, :]
            Y_before_pca = Y_before_pca[:max_to_use, :]
            Y = model.pca.inverse_transform(Y_before_pca)

            print "--> Now running the model on this data"
            if feature_type == 'mahalanobis':

                cov = np.dot(model.pca.components_, model.pca.components_.T)
                print "Cov is ", cov.shape
                distances = pairwise_distances(
                    model.training_Y, Y=Y_before_pca, metric='mahalanobis', VI=cov, n_jobs=-1)
                print "distances is ", distances.shape

                # now find the location of the minimum
                best_matches = np.argmin(distances, axis=0)

                predictions = []
                for match in best_matches:
                    full_pred = model.pca.inverse_transform(model.training_Y[match])
                    temp_mask = model.masks_pca.inverse_transform(model.training_masks[match])
                    predictions.append((full_pred, temp_mask))

            elif feature_type == 'gt_oracle_full':
                # do distances to each training example in turn
                # each row is a training example, each column a test example
                distances = np.zeros((model.training_Y.shape[0], max_to_use))

                for start_idx in np.arange(0, model.training_Y.shape[0], 10000):
                    training_y = model.training_Y[start_idx:(start_idx+10000)]
                    full_training_y = model.pca.inverse_transform(training_y)

                    dists = pairwise_distances(full_training_y, Y, n_jobs=-1)
                    distances[start_idx:(start_idx+10000), :] = dists

                    if np.mod(start_idx, 10000) == 0:
                        print "Done %d of %d" %(start_idx, model.training_Y.shape[0])

                # now find the location of the minimum
                best_matches = np.argmin(distances, axis=0)

                predictions = []
                for match in best_matches:
                    full_pred = model.pca.inverse_transform(model.training_Y[match])
                    temp_mask = model.masks_pca.inverse_transform(model.training_masks[match])
                    predictions.append((full_pred, temp_mask))

            elif feature_type == 'gt_oracle':
                nbrs = NearestNeighbors(
                    n_neighbors=1, algorithm='kd_tree').fit(model.training_Y)
                _, indices = nbrs.kneighbors(Y_before_pca)

                temp_predictions = np.squeeze(model.pca.inverse_transform(
                    model.training_Y[indices]))
                temp_masks = np.squeeze(model.masks_pca.inverse_transform(
                    model.training_masks[indices]))

                predictions = []
                for p, m in zip(temp_predictions, temp_masks):
                    predictions.append((p, m))
            else:
                predictions = [model.predict(x) for x in X]

            # print "--> Rendering some results"
            # if vox_type == 'tall':
            #     S = (20, 40, 50)
            # else:
            #     S = (20, 40, 20)

            # savefolder = '/tmp/renders/' + feature_type + '_' + vox_type + '/'
            # if not os.path.exists(savefolder):
            #     os.makedirs(savefolder)
            # for idx in range(10):
            #     savepath = savefolder + '%05d_gt.png' % idx
            #     rendering.render_single_voxlet(Y[idx].reshape(S),
            #         savepath, height=vox_type)
            #     savepath = savefolder + '%05d_pred.png' % idx
            #     rendering.render_single_voxlet(predictions[idx][0].reshape(S),
            #         savepath, height=vox_type)

            #     print "--> Now rendering slices..."
            #     arrs = (Y[idx].reshape(S),
            #         predictions[idx][0].reshape(S),
            #         predictions[idx][1].reshape(S))
            #     rendering.plot_slices(
            #         arrs, savefolder + '%05d_slices.png' % idx,
            #         ismask=[0, 0, 1])


            print "--> Evaluating (SHOULD only do on the voxels in the mask...)"
            # getting the average prediction of all the preidctions....
            name = vox_type + ' ' + feature_type
            this_evals = [evaluate(y, prediction, prediction_mask)
                for y, (prediction, prediction_mask) in zip(Y, predictions)]
            temp = {}
            for score in scores:
                temp[score] = np.array([tt[score] for tt in this_evals]).mean()
            results[name] = temp


    # Printing the table nicely

    print '\n'
    print ' ' * 25,
    for score in scores:
        print score.ljust(10),
    print '\n' + '-' * 55

    for name, result in results.iteritems():
        print name.ljust(25),
        for score_type in scores:
            score = result[score_type]
            print ('%0.4f' % score).ljust(10),
        print ""