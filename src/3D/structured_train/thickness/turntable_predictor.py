'''
This is a class for dense prediction of turntable objects
It will be given the trained forest
It will be given a depth image
It can be given the GT thickness
It will do things like:
- create prediction image
- create prediction voxels
- evaluate ROC curves
- plot slices through the voxels

'''

import matplotlib
#matplotlib.use('pdf')
import matplotlib.pyplot as plt

import numpy as np
import cPickle as pickle
import scipy.io
import os

import paths
import combine_data
import compute_data
import images
import displaying as disp

class TurntablePredictor(object):

    def __init__(self, forest):
        self.nan_replacement_value = 5.0
        self.forest = forest
        self.feature_engine = compute_data.DepthFeatureEngine()


    def set_image(self, im):
        '''set the rgbd data, should really be of class RgbdImage or descended'''
        self.im = im
        self.feature_engine.set_image(im)

    def set_forest(self, forest):
        self.forest = forest
#pickle.load(open(paths.rf_folder_path + forest_name + '.pkl'))

    def compute_features(self):
        '''compute dense features across the whole image'''
        self.feature_engine.dense_sample_from_mask()#1000)
        #self.feature_engine.random_sample_from_mask(10)
        self.feature_engine.compute_features_and_depths(jobs=1)
        self.all_features = self.feature_engine.features_and_depths_as_dict()

    def predict_per_tree(self, random_forest, X):
        return np.array([tree.predict(X) for tree in random_forest.estimators_])

    def classify_features(self):

        ''' extracting the features'''
        patch = np.array(self.all_features['patch_features'])
        patch[np.isnan(patch)] = self.nan_replacement_value
        patch[patch > 0.1]  = self.nan_replacement_value

        spider = np.array(self.all_features['spider_features'])
        spider = combine_data.replace_nans_with_col_means(spider)

        #import pdb; pdb.set_trace()
        print spider.shape
        print patch.shape

        X = np.concatenate((patch, spider), axis=1)

        ''' classifying'''
        self.tree_predictions = self.predict_per_tree(self.forest, X)
        self.Y_pred = np.median(self.tree_predictions, axis=0)
        #self.Y_pred = np.sum(np.isnan(np.array(self.all_features['patch_features'])), axis=1)
        #self.Y_pred_rf = rf.predict(X)

        ''' reconstructing the image'''
        test_idxs = np.array(self.all_features['indices']).T

        print self.Y_pred.shape

        self.prediction_image = disp.reconstruct_image(test_idxs, self.Y_pred, (480, 640))

        # doing GT...
        self.Y_gt = np.array(self.all_features['depth_diffs'])
        self.GT_image = disp.reconstruct_image(test_idxs, np.array(self.Y_gt), (480, 640))


    def prediction_gt_image(self):
        return disp.crop_concatenate((self.GT_image, self.prediction_image), 10)

    def predictions_as_dict(self):
        return dict(pred_image=self.prediction_image,
                    Y_pred=self.Y_pred,
                    tree_predictions=self.tree_predictions)

    # def plot_prediction_by_GT(self):


    # def plot_prediction(self):



    # def reconstruct_volume(self):


    # def plot_slice(self):

overwrite = False

if __name__ == '__main__':

    print "Loading image..."
    im = images.MaskedRGBD()
    im.load_bigbird("coffee_mate_french_vanilla", "NP1_150")
    im.compute_edges_and_angles(edge_threshold = 1.0)
    im.mask = im.depth < 0.84
    im.mask[:180, :] = 0
    im.mask[400:, :] = 0
    im.mask[:, :200] = 0
    im.mask[:, 380:] = 0

    #print im.mask.dtype, im.depth.dtype
    #im.depth[im.mask==1] = 0
    # plt.imshow(im.depth)#rgb[:, :, 1] + 100*im.depth)
    # plt.colorbar()
    # plt.show()
    # returnsds

    # loading the saved forest
    print "Loading forest..."
    rf = pickle.load( open(paths.rf_folder_path + "patch_spider_5k_10trees.pkl", "r") )

    print "Creating predictor"
    pred = TurntablePredictor(rf)
    pred.set_image(im)

    print "Computing and classifying features"
    pred.compute_features()
    pred.classify_features()
    plt.imshow(pred.prediction_image, interpolation='nearest')
    plt.colorbar()
    plt.show()

    #im.disp_channels()







