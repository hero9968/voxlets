'''
This is a class for dense prediction
It will be given the trained forest
It will be given a depth image
It can be given the GT thickness
It will do things like:
- create prediction image
- create prediction voxels
- evaluate ROC curves
- plot slices through the voxels

decide what to do about compute_data using the filename as init

two things to do to fix:
1) Change compute data so has an internal 'set depth images' func 
2) Change this file so it uses names instead of images.
'''

import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt

import numpy as np
import cPickle as pickle
import scipy.io
import os

import paths
import combine_data
import images
import compute_data
import displaying as disp

class DensePredictor(object):

    def __init__(self, forest):

        self.nan_replacement_value = 5.0
        self.feature_engine = compute_data.DepthFeatureEngine()


        # load the forest to use
        #self.load_forest(forest_to_use)
        self.forest = forest


    def set_image(self, im):
        # creating the engine for the feature computation
        self.im = im
        self.feature_engine.set_image(im)


    def compute_features(self):
        '''compute dense features across the whole image'''

        self.feature_engine.dense_sample_from_mask()
        self.feature_engine.compute_features_and_depths()
        self.all_features = self.feature_engine.features_and_depths_as_dict()


    def predict_per_tree(self, random_forest, X):
        return np.array([tree.predict(X) for tree in random_forest.estimators_])


    def classify_features(self):

        ''' extracting the features'''
        patch = np.array(self.all_features['patch_features'])
        patch[np.isnan(patch)] = self.nan_replacement_value

        spider = np.array(self.all_features['spider_features'])
        spider = combine_data.replace_nans_with_col_means(spider)

        X = np.concatenate((patch, spider), axis=1)

        ''' classifying'''
        self.tree_predictions = self.predict_per_tree(self.forest, X)
        self.Y_pred = np.median(self.tree_predictions, axis=0)
        #self.Y_pred_rf = rf.predict(X)

        ''' reconstructing the image'''
        test_idxs = np.array(self.all_features['indices']).T

        print self.Y_pred.shape

        self.prediction_image = disp.reconstruct_image(test_idxs, self.Y_pred, (240, 320))

        # doing GT...
        self.Y_gt = np.array(self.all_features['depth_diffs'])
        self.GT_image = disp.reconstruct_image(test_idxs, np.array(self.Y_gt), (240, 320))


    def prediction_gt_image(self):
        return disp.crop_concatenate((self.GT_image, self.prediction_image), 10)

    def predictions_as_dict(self):
        return dict(GT_image=pred.GT_image,
                    pred_image=pred.prediction_image,
                    Y_gt=pred.Y_gt,
                    Y_pred=pred.Y_pred,
                    tree_predictions=self.tree_predictions)            

    # def plot_prediction_by_GT(self):


    # def plot_prediction(self):



    # def reconstruct_volume(self):


    # def plot_slice(self):

overwrite = False

if __name__ == '__main__':

    # loading the saved forest
    print "Loading forest..."
    rf = pickle.load( open(paths.rf_folder_path_small + "patch_spider_5k_10trees.pkl", "r") )

    #'2566f8400d964be69c3cb90632bf17f3' #
    #modelname = '109d55a137c042f5760315ac3bf2c13e'

    savefolder = paths.dense_savefolder

    # prediction object - its job is to do the prediction!
    f = open(paths.test_path, 'r')
    object_names = [l.strip() for l in f]
    
    for idx, modelname in enumerate(object_names):

        print "Processing " + modelname + ", number: " + str(idx)

        modelsavefolder = savefolder + modelname + "/"
        if not os.path.exists(modelsavefolder):
            os.makedirs(modelsavefolder)

        for view in paths.views:

            savepath = modelsavefolder + str(view) + '.mat'
            imgsavepath = modelsavefolder + str(view) + '.pdf'

            if os.path.exists(savepath) and not overwrite:
                print "Skipping " + modelname + " " + str(view) + " as it exists already"

            print "Loading image"
            im = images.CroppedRGBD()
            im.load_bigbird_from_mat(modelname, view)

            print "Computing features"
            pred = DensePredictor(rf)
            pred.set_image(im)
            pred.compute_features()

            print "Doing prediction"
            pred.classify_features()

            print "Extracting and saving results"
            d = pred.predictions_as_dict()
            scipy.io.savemat(savepath, d)

            print "Creating image"
            plt.clf()
            plt.imshow(pred.prediction_gt_image())
            plt.savefig(imgsavepath)

            print "Done..."

    print "Done all"


        #plt.imshow(pred.prediction_image)






