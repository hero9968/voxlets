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
import compute_data
import displaying as disp

import voxel_data

class DensePredictor(object):

    def __init__(self, forest):

        self.nan_replacement_value = 5.0


        # load the forest to use
        #self.load_forest(forest_to_use)
        self.forest = forest


    def load_renders(self, modelname, view_idx):
        # creating the engine for the feature computation
        self.feature_engine = compute_data.DepthFeatureEngine(modelname, view_idx+1)


    def set_forest(self, forest):
        ''' sets the sklearn forest object'''
        self.rf = forest

    def compute_features(self, slice_idx=None):
        '''compute dense features across the whole image OR for a single slice'''

        self.slice_idx = slice_idx

        # sampling from just the slices
        all_idxs = np.array(np.nonzero(self.feature_engine.mask)).T
        #idxs = [all_idxs[0], all_idxs[1] if all_idxs[0] == slice_idx]
        idxs = np.array([[t0, t1] for t0, t1 in all_idxs if t0==slice_idx])
        self.feature_engine.indices = idxs
        print "Shape: " + str(idxs.shape)
        print all_idxs
        self.feature_engine.samples_per_image = idxs.shape[0]
        self.idxs_used = idxs

        self.feature_engine.compute_features_and_depths(jobs=1)
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

        self.Y_gt = np.array(self.all_features['depth_diffs'])
        self.GT_image = disp.reconstruct_image(test_idxs, np.array(self.Y_gt), (240, 320))


        print "Sizes are " + str(self.idxs_used.shape) + " " + str(self.Y_pred.shape)
        slice_idxs = np.copy(self.idxs_used).T
        slice_idxs[0, :] = 0
        print slice_idxs
        self.prediction_slice = disp.reconstruct_image(slice_idxs, self.Y_pred, (1, 320))

        self.Y_gt = np.array(self.all_features['depth_diffs'])
        self.GT_slice = self.GT_image[self.slice_idx, :]

        print "Created slice of size " + str(self.prediction_slice.shape)
        print "Created slice of size " + str(self.GT_slice.shape)

        

    def prediction_gt_image(self):
        return disp.crop_concatenate((self.GT_image, self.prediction_image), 10)


    def fill_slice(self, slice_idx):
        '''
        populates a 2D slice through the voxel volume 
        '''
        filler = voxel_data.SliceFiller(front_slice, back_slice_predictions, focal_length)
        return filler.fill_slice(0, 2)



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

overwrite = True


if __name__ == '__main__':

    # loading the saved forest
    print "Loading forest..."
    rf = pickle.load( open(paths.rf_folder_path + "patch_spider_5k_10trees.pkl", "r") )

    #'2566f8400d964be69c3cb90632bf17f3' #
    #modelname = '109d55a137c042f5760315ac3bf2c13e'

    savefolder = paths.base_path + "structured/dense_predictions/"

    # prediction object - its job is to do the prediction!

    object_names = [l.strip() for l in scipy.io.loadmat(paths.split_path)['train_names']]
    
    for idx, modelname in enumerate(object_names[:3]):

        print "Processing " + modelname + ", number: " + str(idx)

        modelsavefolder = savefolder + modelname + "/"
        if not os.path.exists(modelsavefolder):
            os.makedirs(modelsavefolder)

        view_idx = 12

        savepath = modelsavefolder + str(view_idx) + '_slices.mat'
        imgsavepath = modelsavefolder + str(view_idx) + '_slices.pdf'

        if os.path.exists(savepath) and not overwrite:
            print "Skipping " + modelname + " " + str(view_idx)

        print "Computing features"
        pred = DensePredictor(rf)
        pred.load_renders(modelname, view_idx)

        plt.subplot(2, 2, 1)
        plt.imshow(pred.feature_engine.frontrender)

        for idx, slice_idx in enumerate([130, 120, 180]):
            print "Computing and classifying all points in slice" + str(slice_idx)
            pred.compute_features(slice_idx)
            pred.classify_features()

            print "Creating slice"
            slice_image = pred.fill_slice(slice_idx)


            print "Adding to plot"
            plt.subplot(2, 2, idx+1)
            plt.imshow(slice_image)

        plt.show()
        
        #plt.savefig(imgsavepath)

        print "Done..."

        print "Breaking"
        break

    print "Done all"


        #plt.imshow(pred.prediction_image)






