'''
Combines all the computed features into one file. This makes loading quicker when training the models
'''

import numpy as np
import os
import scipy.io
import collections
import scipy.stats as stats
import paths
import cPickle as pickle
import random
# User options



def load_modeldata(modelname):
	modelpath = paths.model_features + modelname + '.mat'

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
	col_mean = stats.nanmean(X,axis=0)
	col_mean = nan_to_value(col_mean, 0)
	inds = np.where(np.isnan(X))
	X[inds]=np.take(col_mean,inds[1])
	return X

features_to_sample_per_object = 1000

 # small model has very few features, just used for testing algorithms...
for small_model in [True, False]:
	for category in ['train', 'test']:  # options are test and train - are we doing test or train data?

		# setting paths
		combined_features_save_path = paths.base_path + 'structured/combined_features/'

		combined_features_save_path += category  # should be 'test' or 'train'
		if small_model:	combined_features_save_path += '_small'
		combined_features_save_path += '.pkl'

		# loading the data
		# For now, am only going to use one file to train on...
		object_names = scipy.io.loadmat(paths.split_path)[category + '_names']

		print "There are " + str(len(object_names)) + " objects"

		for idx, line in enumerate(object_names):

			print "Loading " + str(idx) + ": " + line
			temp = load_modeldata(line.strip())

			all_idxs = xrange(temp['depth_diffs'].shape[0])
			to_use = random.sample(all_idxs, features_to_sample_per_object)

			if idx == 0:
				patch_features_nan = np.array(temp['patch_features'][to_use, :])
				spider_features = np.array(temp['spider_features'][to_use, :])
				Y = np.array(temp['depth_diffs'][to_use, :])

			elif temp:
				patch_features_nan = np.append(patch_features_nan, temp['patch_features'][to_use, :], axis=0)
				spider_features = np.append(spider_features, temp['spider_features'][to_use, :], axis=0)
				Y = np.append(Y, temp['depth_diffs'][to_use, :], axis=0)

			if small_model and idx > 5:
				break
		
		# constructing the X and Y variables
		Y = nan_to_value(Y, 0)

		print "Converting patch features... size is " + str(patch_features_nan.shape)
		patch_features = nan_to_value(patch_features_nan, 0).astype(np.float16)


		print "Converting spider features... size is " + str(spider_features.shape)
		spider_features = replace_nans_with_col_means(spider_features).astype(np.float16)
		print "Nan count: " + str(np.sum(np.isnan(spider_features)))

		print "Size of Y is " + str(Y.shape)

		print "Saving to file..."
		d = dict(spider_features=spider_features,
				 patch_features=patch_features,
				 Y=Y)
		f = open(combined_features_save_path,'wb')
		pickle.dump(d,f)
		f.close()
		#scipy.io.savemat(combined_features_save_path, d)

		print "Done..."

print "Done all!"