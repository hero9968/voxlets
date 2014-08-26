'''
This script brings together all the structured information and trains a RF on them
'''
#import pdb; pdb.set_trace()
import os
import collections
import numpy as np
import scipy.io
import scipy.stats as stats
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
import cPickle as pickle
import timeit

base_path = os.path.expanduser("~/projects/shape_sharing/data/3D/basis_models/")
models_list = base_path + 'databaseFull/fields/models.txt'
split_path = base_path + 'structured/split.mat'

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
		print idx
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

def resample_inputs(X, Y, num_samples):
	'''
	returns exactly num_samples items from X and Y. 
	Samples without replacement if num_samples < number of data points
	'''
	assert(X.shape[0]==len(Y))

	if len(Y) < num_samples:
		# sample with replacement
		idxs = np.random.randint(0, len(Y), num_samples)
	else:
		# sample without replacment
		idxs = np.random.permutation(len(Y))[0:num_samples]

	return X[idxs,:], Y[idxs]

# setting up the options
number_trees = 10
num_samples = 250000
use_spider = False
testing_only = False # if true, then only load a few of the models (v quick, good for testing)

if __name__ == '__main__':

	# seeding random numbers so that in theory each run will generate the same forest 
	# - not really sure if this will be the case though
	np.random.seed(1)

	# loading the data
	# For now, am only going to use one file to train on...
	#f = open(models_list, 'r')
	train_names = scipy.io.loadmat(split_path)['train_names']
	rawdata = []
	print "There are " + str(len(train_names)) + " training objects"
	for idx, line in enumerate(train_names):
		temp = load_modeldata(line.strip())
		if temp:
			rawdata.append(temp)
		print idx 
		if testing_only and idx > 7:
			break

	data = list_of_dicts_to_dict(rawdata)
	print "Spider is " + str(np.array(data['spider_features']).shape)

	# constructing the X and Y variables
	Y = np.array(data['depth_diffs']).ravel()
	Y = nan_to_value(Y, 0)

	print "Reshaping..."
	patch_feature_dimension = data['patch_features'][0].shape[1]
	patch_feature_nan = np.array(data['patch_features']).reshape(-1, patch_feature_dimension)

	print "Removing nans..."
	patch_features = nan_to_value(patch_feature_nan, 0)

	if use_spider:
		spider_feature_dimension = data['spider_features'][0].shape[1]
		spider_features = replace_nans_with_col_means(np.array(data['spider_features']))
		spider_features = spider_features.reshape(-1, spider_feature_dimension)

	X = patch_features
	#X = np.concatenate((patch_features, spider_features), axis=1)

	print "Before resampling..."
	print "X shape is " + str(X.shape)
	print "Y shape is " + str(Y.shape)

	# Here do some subsampling to reduce the size of the input datasets...
	X, Y = resample_inputs(X, Y, num_samples)

	print "After resampling..."
	print "X shape is " + str(X.shape)
	print "Y shape is " + str(Y.shape)

	# training the forest
	print "Training forest..."
	tic = timeit.default_timer()
	clf = RandomForestRegressor(n_estimators=number_trees,n_jobs=-1)
	clf.fit(X,Y)
	pickle.dump(clf, open("spidermodel2.pkl", "wb") )
	print 'Done training in ' + str(timeit.default_timer() - tic)
