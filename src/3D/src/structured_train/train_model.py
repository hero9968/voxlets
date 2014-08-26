'''
This script brings together all the structured information and trains a RF on them
'''

#import pdb; pdb.set_trace()
import os
import numpy as np
import scipy.io
from sklearn.ensemble import RandomForestRegressor
import cPickle as pickle
import timeit

small_model = True # if true, then only load a few of the models (v quick, good for testing)

base_path = os.path.expanduser("~/projects/shape_sharing/data/3D/basis_models/")
models_list = base_path + 'databaseFull/fields/models.txt'

if small_model:
	combined_features_path = base_path + 'structured/combined_features_small.mat'
	num_samples = 2500
else:
	combined_features_path = base_path + 'structured/combined_features.mat'
	num_samples = 250000


def resample_inputs(X, Y, num_samples):
	'''
	returns exactly num_samples items from X and Y. 
	Samples without replacement if num_samples < number of data points
	'''
	assert(X.shape[0]==len(Y))
	np.random.seed(1)

	if len(Y) < num_samples:
		# sample with replacement
		idxs = np.random.randint(0, len(Y), num_samples)
	else:
		# sample without replacment
		idxs = np.random.permutation(len(Y))[0:num_samples]

	return X[idxs,:], Y[idxs]


# setting up the options
number_trees = 10
use_spider = False

if __name__ == '__main__':

	# loading the data
	train_data = scipy.io.loadmat(combined_features_path)
	X = train_data['patch_features']
	Y = train_data['Y'].flatten()
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
	clf = RandomForestRegressor(n_estimators=number_trees,n_jobs=-1,random_state=1)
	clf.fit(X,Y)

	print "Saving to disk..."
	pickle.dump(clf, open("spidermodel2.pkl", "wb") )
	print 'Done training in ' + str(timeit.default_timer() - tic)
