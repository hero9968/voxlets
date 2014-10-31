'''
Do prediction for the models but only for the sparse features
Dense prediction (i.e. doing prediction for an entire image) will 
be done elsewhere
'''
import os
import numpy as np
import cPickle as pickle
import scipy.io
from sklearn.metrics import roc_curve, auc
import yaml
import paths

# setting paths
small_dataset = False

if small_dataset:
	combined_features_path = paths.combined_test_features_small
	results_folder = paths.results_folder_small
else:
	combined_features_path = paths.combined_test_features
	results_folder = paths.results_folder

# setting options for the per-voxel ROC curves
# NOTE: Currently the ROC is being computed on the depth-diff measurements, including the max depth
voxel_depth = 100  # how deep to consider the scene to be in voxels
max_depth_to_consider = 0.3 # depths beyond this are ignored. <voxel_depth> pixels span the whole depth
scale_factor = voxel_depth/max_depth_to_consider

def predict_per_tree(random_forest, X):
    return np.array([tree.predict(X) for tree in random_forest.estimators_])

def populate_voxels(prediction_values, voxel_depth):
	'''
	populates an array of length voxel_depth with occupancy values in [0,1], 
	based on the list of prediction values. 
	'''
	pixel_space = np.zeros((voxel_depth,), dtype=float)
	for pred in prediction_values:
		pixel_space[:pred] += 1
	return pixel_space / len(prediction_values)

# loading test data
f = open(combined_features_path, 'rb')
test_data = pickle.load(f)
f.close()
Y_gt = np.array(test_data['Y'])

# doing each model in turn
all_models = yaml.load(open(paths.model_config, 'r'))

for modeloption in all_models:
	
	rfmodel_path = paths.rf_folder_path + modeloption['name'] + '.pkl'

	if not os.path.isfile(rfmodel_path):
		print "Cannot find model: " + rfmodel_path
		continue

	# loading model
	print "Loading model: " + modeloption['name']
	clf = pickle.load(open(rfmodel_path, "r") )

	# loading the correct features to satisfy this model
	X = [test_data[feature] for feature in modeloption['features']]
	X = np.concatenate(X, axis=1)

	# rf prediction
	Y_pred = predict_per_tree(clf, X)
	Y_avg = np.median(Y_pred, axis=0)
	print Y_pred.shape, Y_avg.shape, Y_gt.shape

	# evaluation of result
	per_pixel_squared_error = np.square(Y_gt.flatten() - Y_avg.flatten())
	mean_ssd = np.mean(per_pixel_squared_error)

	print "Max GT depth is " + str(np.max(Y_gt))
	print "Max pred depth is " + str(np.max(Y_pred))
	print "Scale factor is " + str(scale_factor)

	# doing the ROC curves
	scaled_pred = (scale_factor * Y_pred).astype(int).transpose()
	scaled_gt = (scale_factor * Y_gt).astype(int).flatten()
	print "Max GT depth (voxels) is " + str(np.max(scaled_gt))
	print "Max pred depth (voxels) is " + str(np.max(scaled_pred.flatten()))
	
	all_gt_voxels = [populate_voxels([gt_val], voxel_depth) for gt_val in scaled_gt]
	all_pred_voxels = [populate_voxels(pred_val, voxel_depth) for pred_val in scaled_pred]

	all_pred_voxels = np.array(all_pred_voxels).flatten()
	all_gt_voxels = np.array(all_gt_voxels).flatten()

	# computing ROC curve on a per-voxel basis
	fpr, tpr, thresholds = roc_curve(all_gt_voxels, all_pred_voxels)
	roc_auc = auc(fpr, tpr)

	result = dict(Y_gt=Y_gt, 
				Y_pred=Y_pred, 
				per_pixel_squared_error=per_pixel_squared_error, 
				mean_ssd=mean_ssd,
				roc_auc=roc_auc,
				fpr=fpr,
				tpr=tpr,
				thresholds=thresholds,
				all_gt_voxels=all_gt_voxels,
				all_pred_voxels=all_pred_voxels)

	result_savepath = results_folder + modeloption['name'] + '.mat'
	print "Saving to: " + result_savepath
	scipy.io.savemat(result_savepath, result)

	print "Area under the ROC curve : %f" % roc_auc
