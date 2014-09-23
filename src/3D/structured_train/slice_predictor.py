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

class SlicePredictor(object):

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
        #print "Shape: " + str(idxs.shape)
        #print all_idxs
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
        print "Tree predict size is " + str(self.tree_predictions.shape)
        self.Y_pred = np.median(self.tree_predictions, axis=0)
        #self.Y_pred_rf = rf.predict(X)

        ''' reconstructing the image'''
        test_idxs = np.array(self.all_features['indices']).T

        '''reconstructing the gt image'''
        self.Y_gt = np.array(self.all_features['depth_diffs'])
        self.GT_image = disp.reconstruct_image(test_idxs, np.array(self.Y_gt), (240, 320))
        self.GT_slice = self.GT_image[self.slice_idx, :]
        #print self.Y_pred.shape

        print "Sizes are " + str(self.idxs_used.shape) + " " + str(self.Y_pred.shape)

        '''reconstructing the prediction'''
        slice_idxs = np.copy(self.idxs_used).T
        slice_idxs[0, :] = 0
        #print slice_idxs
        self.prediction_slice = disp.reconstruct_image(slice_idxs, self.Y_pred, (1, 320))

        '''reshaping the tree predictions to be the full size of the slice'''
        self.predictions = []
        for tree in self.tree_predictions:
            tree_slice = disp.reconstruct_image(slice_idxs, tree, (1, 320))
            self.predictions.append(tree_slice)
        self.predictions = np.array(self.predictions)

        self.Y_gt = np.array(self.all_features['depth_diffs'])

        self.front_slice = self.feature_engine.frontrender[self.slice_idx, :]

        print "Created slice of size " + str(self.prediction_slice.shape)
        print "Created slice of size " + str(self.GT_slice.shape)

        

    def prediction_gt_image(self):
        return disp.crop_concatenate((self.GT_image, self.prediction_image), 10)


    def fill_slice(self):
        '''
        populates a 2D slice through the voxel volume 
        '''

        filler = voxel_data.SliceFiller(self.front_slice, self.predictions, 570.0/2)
        print "Min is " + str(np.nanmin(self.front_slice))
        print "Max is " + str(np.nanmax(self.predictions))
        return filler.extract_warped_slice(1.0, 2.0, gt=self.GT_slice)


    def predictions_as_dict(self):
        return dict(GT_image=pred.GT_image,
                    pred_image=pred.prediction_image,
                    Y_gt=pred.Y_gt,
                    Y_pred=pred.Y_pred,
                    tree_predictions=self.tree_predictions)            


overwrite = False

if __name__ == '__main__':

    # loading the saved forest
    print "Loading forest..."
    rf = pickle.load( open(paths.rf_folder_path + "patch_spider_1M_10trees.pkl", "r") )

    #'2566f8400d964be69c3cb90632bf17f3' #
    #modelname = '109d55a137c042f5760315ac3bf2c13e'

    savefolder = paths.base_path + "slice_predictions/"

    # prediction object - its job is to do the prediction!

    object_names = [l.strip() for l in scipy.io.loadmat(paths.split_path)['test_names']]
    print "First is " + object_names[0]

    for idx, modelname in enumerate(object_names[1:100]):
        if idx < 21: continue

        print "Processing " + modelname + ", number: " + str(idx)

        view_idx = np.random.choice([5, 12, 15, 20])

        imgsavepath = savefolder + modelname + "_" + str(view_idx) + '_slices.pdf'

        if os.path.exists(imgsavepath) and not overwrite:
            print "Skipping " + modelname + " " + str(view_idx)

        print "Computing features"
        pred = SlicePredictor(rf)
        pred.load_renders(modelname, view_idx)

        print "Choosing the slices "
        this_frontrender = np.copy(pred.feature_engine.frontrender)
        locs = np.nonzero(np.any(~np.isnan(this_frontrender), axis=1))

        object_top = np.min(locs)
        object_bottom = np.max(locs)
        object_height = object_bottom - object_top
        print object_top, object_bottom
        slices = (object_top + object_height * np.array([0.25, 0.5, 0.75])).astype(int)
        print slices

        plt.clf()
        for idx, slice_idx in enumerate(slices):
            print "Computing and classifying all points in slice" + str(slice_idx)
            pred.compute_features(slice_idx)
            pred.classify_features()

            print "Creating slice"
            slice_image = pred.fill_slice()
            print slice_image.shape

            print "Adding to plot"
            plt.subplot(2, 2, idx+2)
            plt.imshow(slice_image)
            plt.gca().invert_yaxis()

            this_frontrender[slice_idx, :] = 0

        plt.subplot(2, 2, 1)
        plt.imshow(this_frontrender)
        plt.title(modelname + " : " + str(view_idx))

        #plt.show()      
        plt.savefig(imgsavepath)
        #plt.savefig('./')

        print "Done..."
        #print "Breaking"
        #break

    print "Done all"
