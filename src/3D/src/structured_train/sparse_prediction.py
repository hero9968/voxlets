'''
Do prediction for the models but only for the sparse features
Dense prediction (i.e. doing prediction for an entire image) will 
be done elsewhere
'''

import os
import numpy as np
import cPickle as pickle
import scipy.io
import matplotlib.pyplot as pl
from sklearn.metrics import roc_curve, auc

# setting paths
base_path = os.path.expanduser("~/projects/shape_sharing/data/3D/basis_models/")
combined_features_path = base_path + 'structured/combined_features/test_small.mat'
rfmodel_path = base_path + '../structured_models/smallmodel.pkl'
testing = True

# setting options for the per-voxel ROC curves
# NOTE: Currently the ROC is being computed on the depth-diff measurements, including the max depth
voxel_depth = 100  # how deep to consider the scene to be in voxels
max_depth_to_consider = 0.25 # depths beyond this are ignored. <voxel_depth> pixels span the whole depth
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
test_data = scipy.io.loadmat(combined_features_path)
X = np.array(test_data['patch_features'])
Y_gt = np.array(test_data['Y'])

# loading model
clf = pickle.load(open(rfmodel_path, "r") )

# rf prediction
Y_pred = predict_per_tree(clf, X)
Y_avg = np.median(Y_pred, axis=0)

# evaluation of result
per_pixel_squared_error = np.square(Y_gt - Y_avg)
mean_ssd = np.mean(per_pixel_squared_error)

print "Max GT depth is " + str(np.max(Y_gt))
print "Max pred depth is " + str(np.max(Y_pred))
print "Scale factor is " + str(scale_factor)

# doing the ROC curves
scaled_pred = (scale_factor * Y_pred).astype(int).flatten()
scaled_gt = (scale_factor * Y_gt).astype(int).transpose()
all_gt_voxels = [populate_voxels([gt_val], voxel_depth) for gt_val in scaled_gt]
all_pred_voxels = [populate_voxels(pred_val, voxel_depth) for pred_val in scaled_pred]

print np.mean(np.array(all_gt_voxels), axis=0)
print np.max(scaled_gt)

all_pred_voxels = np.array(all_pred_voxels).flatten()
all_gt_voxels = np.array(all_gt_voxels).flatten()

# computing ROC curve on a per-voxel basis
fpr, tpr, thresholds = roc_curve(all_gt_voxels, all_pred_voxels)
roc_auc = auc(fpr, tpr)

result = dict(Y_gt=Y_gt, 
			Y_pred=Y_pred, 
			per_pixel_squared_error=per_pixel_squared_error, 
			mean_ssd=mean_ssd,
			all_gt_voxels=all_gt_voxels,
			all_pred_voxels=all_pred_voxels)

scipy.io.savemat('tempresult.mat', result)


print "Area under the ROC curve : %f" % roc_auc

# Plot ROC curve
# (Really this needs to be done for all models at a later time...)
pl.clf()
pl.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
pl.plot([0, 1], [0, 1], 'k--')
pl.xlim([0.0, 1.0])
pl.ylim([0.0, 1.0])
pl.xlabel('False Positive Rate')
pl.ylabel('True Positive Rate')
pl.title('Receiver operating characteristic example')
pl.legend(loc="lower right")
pl.savefig('roc.eps')
#pl.show()



# for pixel_pred, pixel_GT in zip(scaled_pred.transpose(), scaled_gt.flatten()):

# 	# populate the pixel space
# 	pixel_space = np.zeros((voxel_depth,), dtype=int)
# 	for pred in pixel_pred:
# 		pixel_space[:pred] += 1

# 	gt_space = np.zeros((voxel_depth,), dtype=int)
# 	gt_space[:pixel_GT] = 1

# 	all_pred_voxels.append(pixel_space)
# 	all_gt_voxels.append(gt_space) 
#number_trees = scaled_pred.shape[0]
#all_pred_voxels = np.array(all_pred_voxels, dtype=float).flatten() / number_trees
