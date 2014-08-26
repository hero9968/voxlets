'''
Combines all the computed features into one file. This makes loading quicker when training the models
'''

import numpy as np
import os
import scipy.io
import collections
import scipy.stats as stats

# User options
small_model = True # small model has very few features, just used for testing algorithms...
category = 'train'  # options are test and train - are we doing test or train data?

# setting paths
base_path = os.path.expanduser("~/projects/shape_sharing/data/3D/basis_models/")
split_path = base_path + 'structured/split.mat'
combined_features_save_path = base_path + 'structured/combined_features/'

combined_features_save_path += category  # should be 'test' or 'train'
if small_model:	combined_features_save_path += '_small'
combined_features_save_path += '.mat'

def load_modeldata(modelname):
	modelpath = base_path + 'structured/features/' + modelname + '.mat'
	if os.path.isfile(modelpath):
		return scipy.io.loadmat(modelpath)
	else:
		return []


def list_of_dicts_to_dict(list_of_dicts):
	'''
	http://stackoverflow.com/questions/11450575/
	how-do-i-convert-a-list-of-dictionaries-to-a-dictionary-of-lists-in-python
	'''
	result = collections.defaultdict(list)

	for idx, d in enumerate(list_of_dicts):
		for k, v in d.items():
			result[k].append(v)

	return result

def nan_to_value(X, newval):
	X[np.isnan(X)] = newval
	return X

def replace_nans_with_col_means(X):
	'''
	http://stackoverflow.com/questions/18689235/numpy-array-replace-nan-values-with-average-of-columns
	'''
	print X.shape
	col_mean = stats.nanmean(X,axis=0)
	inds = np.where(np.isnan(X))
	X[inds]=np.take(col_mean,inds[1])
	return X


# loading the data
# For now, am only going to use one file to train on...
object_names = scipy.io.loadmat(split_path)[category + '_names']
rawdata = []
print "There are " + str(len(object_names)) + " objects"
for idx, line in enumerate(object_names):
	print "Loading " + str(idx)
	temp = load_modeldata(line.strip())
	if temp:
		rawdata.append(temp)
	if small_model and idx > 5:
		break

data = list_of_dicts_to_dict(rawdata)
print "Spider is " + str(np.array(data['spider_features']).shape)

# constructing the X and Y variables
Y = np.array(data['depth_diffs']).ravel()
Y = nan_to_value(Y, 0)

print "Extracting patch features..."
patch_feature_dimension = data['patch_features'][0].shape[1]
patch_feature_nan = np.array(data['patch_features']).reshape(-1, patch_feature_dimension)
patch_features = nan_to_value(patch_feature_nan, 0)

print "Extracting spider features..."
spider_feature_dimension = data['spider_features'][0].shape[1]
spider_features = spider_features.reshape(-1, spider_feature_dimension)
spider_features = replace_nans_with_col_means(np.array(data['spider_features']))

print "Saving to file..."
d = dict(spider_features=spider_features,
		 patch_features=patch_features,
		 Y=Y)
scipy.io.savemat(combined_features_save_path, d)

