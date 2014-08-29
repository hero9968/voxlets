'''
Module for loading and saving of data related to the structured prediction project
'''

import paths
import cPickle as pickle
import scipy.io

def load_model(modelname):
	'''
	Loads a specific saved trained model
	'''
	modelpath = paths.rf_folder_path + modelname + '.pkl'
	return pickle.load(open(modelpath, "r") )

def load_combined_features(testtrain, small=False):
	if testtrain=='test' and small==False:
		featurepath = paths.combined_test_features
	elif testtrain=='test' and small==True:
		featurepath = paths.combined_test_features_small
	elif testtrain=='train' and small==False:
		featurepath = paths.combined_train_features
	elif testtrain=='train' and small==True:
		featurepath = paths.combined_train_features_small

	return scipy.io.loadmat(featurepath)

def load_object_names(testtrain):
	'''
	returns a list of names of objects, either test or training...
	testtrain should be test or train 
	Shan't bother doing a 'small' option, but the first 7 (?)
	in each category are from the small set
	'''
	if testtrain == 'all':
		object_names = [l.strip() for l in open(paths.models_list, 'r')]
	else:
		toload = testtrain + '_names'
		object_names = [l.strip() for l in scipy.io.loadmat(paths.split_path)[toload]]
	return object_names

def load_normals(modelname, view_idx):
	'''
	view idx in [1, 42]
	'''
	combined_path = paths.base_path + 'combined_renders/' + modelname + '.mat'
	print "Loading from " + combined_path
	temp = scipy.io.loadmat(combined_path)
	print len(temp['renders']['normals'])
	return temp['renders'][view_idx-1]['normals']