'''

'''

import yaml
import scipy.io
import matplotlib.pyplot as pl
import os.path

results_folder = "./data/results/"
model_config_filepath = './models_config.yaml'

# loading all the model results
all_models = yaml.load(open(model_config_filepath, 'r'))
predictions = []

for modeloption in all_models:
	result_path = results_folder + modeloption['name'] + '.mat'
	if os.path.isfile(result_path):
		this_result = scipy.io.loadmat(result_path)
		this_result['name'] = modeloption['name']
		predictions.append(this_result)

print "In total there are : " + str(len(predictions))
# Plot ROC curve for the models
pl.clf()
for pred in predictions:
	label = pred['name'] + ' (area = %0.2f)' % pred['roc_auc']
	print pred['fpr']
	print pred['tpr']
	print pred['thresholds'].shape
	pl.plot(pred['fpr'].flatten(), pred['tpr'].flatten(), label=label)
	
pl.plot([0, 1], [0, 1], 'k--')
pl.xlim([0.0, 1.0])
pl.ylim([0.0, 1.0])
pl.xlabel('False Positive Rate')
pl.ylabel('True Positive Rate')
pl.title('Receiver operating characteristic example')
pl.legend(loc="lower right") 
#pl.show()
pl.savefig('plots/roc.eps')



# # for pixel_pred, pixel_GT in zip(scaled_pred.transpose(), scaled_gt.flatten()):

# # 	# populate the pixel space
# # 	pixel_space = np.zeros((voxel_depth,), dtype=int)
# # 	for pred in pixel_pred:
# # 		pixel_space[:pred] += 1

# # 	gt_space = np.zeros((voxel_depth,), dtype=int)
# # 	gt_space[:pixel_GT] = 1

# # 	all_pred_voxels.append(pixel_space)
# # 	all_gt_voxels.append(gt_space) 
# #number_trees = scaled_pred.shape[0]
# #all_pred_voxels = np.array(all_pred_voxels, dtype=float).flatten() / number_trees
