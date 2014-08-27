'''
A script to create a train test split.
Will probably save a mat file with a list of training names, testing names and
also a boolean array with the training items indicated
'''
import os
import numpy as np
import scipy.io
import paths

'''
the following is commented out to prevent the accidently overwriting 
of the train/test split if this script is run in error
'''
#split_path = paths.split_path
total_number = 1600

# setting up the options for creating the split
to_use = np.array(range(0, total_number))
test_fraction = 0.2
train_fraction = 0.5

assert test_fraction + train_fraction <= 1

# making the assignments
num_to_use = len(to_use)
number_test = int(test_fraction * num_to_use)
number_train = int(train_fraction * num_to_use)

sorted_list = np.random.permutation(num_to_use)
train_idxs = to_use[sorted_list[:number_train]]
test_idxs = to_use[sorted_list[number_train:(number_train+number_test)]]

# now doing as boolean arrays
train_binary = np.zeros((total_number), dtype=bool)
train_binary[train_idxs] = True
test_binary = np.zeros((total_number), dtype=bool)
test_binary[test_idxs] = True

# creating the train and test names...
f = open(paths.models_list, 'r')
train_names = []
test_names = []
for idx, line in enumerate(f):
	if idx in test_idxs:
		test_names.append(line.strip())
	elif idx in train_idxs:
		train_names.append(line.strip())
f.close()

#print test_names
print len(train_names)
print len(test_names)

# some checks on the data
#print test_idxs.shape
#print train_idxs.shape

# saving the train and test idxs to disk...
split_dict = dict(	test_idxs=test_idxs, 
					train_idxs=train_idxs, 
					test_binary=test_binary, 
					train_binary=train_binary,
					test_names=test_names,
					train_names=train_names)
print "Saving to " + split_path
scipy.io.savemat(split_path, split_dict)

# for later...
#test_views = [0, 5, 10, 15, 20, 25, 30, 35, 40]
#training_views = [0, 5, 10, 15, 20, 25, 30, 35, 40]