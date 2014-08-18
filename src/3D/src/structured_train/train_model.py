'''
This script brings together all the structured information and trains a RF on them
'''

import os
import collections
import numpy as np
import scipy.io
import scipy.stats as stats
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
import cPickle as pickle

base_path = os.path.expanduser("~/projects/shape_sharing/data/3D/basis_models/")
models_list = base_path + 'databaseFull/fields/models.txt'

def load_modeldata(modelname):
	modelpath = base_path + 'features/' + modelname + '.mat'
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


# loading the data
# For now, am only going to use one file to train on...
f = open(models_list, 'r')
rawdata = []
for idx, line in enumerate(f):
	temp = load_modeldata(line.strip())
	if temp:
		rawdata.append(temp)
	print idx 
	if idx > 100:
		break
f.close()

#import pdb; pdb.set_trace()
#
#		modelname = line.strip()
#modelnames = ['109d55a137c042f5760315ac3bf2c13e']
#rawdata = [load_modeldata(name.strip()) for name in f]
#print rawdata
data = list_of_dicts_to_dict(rawdata)

# constructing the X and Y variables
Y = np.array(data['depth_diffs']).ravel()
Y = nan_to_value(Y, 0)
print Y.shape
patch_in = np.array(data['patch_features']).reshape((-1, data['patch_features'][0].shape[1]))
patch_features = nan_to_value(patch_in, 0)
spider_features = replace_nans_with_col_means(np.array(data['spider_features']))
X = patch_features#np.concatenate((patch_features, spider_features), axis=1)
#import pdb; pdb.set_trace()

print "X shape is " + str(X.shape)
print "Y shape is " + str(Y.shape)

# Here do some subsampling to reduce the size of the input datasets...
num_samples = 250000
X,Y = resample_inputs(X, Y, num_samples)

print "X shape is " + str(X.shape)
print "Y shape is " + str(Y.shape)

# training the forest
print "Training forest..."
clf = RandomForestRegressor(n_estimators=10,n_jobs=4)
clf.fit(X,Y)
pickle.dump(clf, open("model.pkl", "wb") )

print clf.feature_importances_

#import pdb; pdb.set_trace()