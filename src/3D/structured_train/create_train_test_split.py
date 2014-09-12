'''
A script to create a train test split.
Will probably save a mat file with a list of training names, testing names and
also a boolean array with the training items indicated
'''
import os
import numpy as np
import scipy.io
import paths
from sklearn.cross_validation import train_test_split

'''
the following is commented out to prevent the accidently overwriting 
of the train/test split if this script is run in error
'''
split_path = paths.split_path
total_number = 1600

# doing a very simple split. Hashes should give a random distribution.
test_idxs = range(0, 300)
train_idxs = range(300, 1100)

# # setting up the options for creating the split
# to_use = np.array(range(0, total_number))
# total_test_fraction = 0.2
# train_fraction = 0.5

# # first 20 objects must be test, for regularity.
# num_fixed_test = 25
# random_test_fraction = total_test_fraction - (num_fixed_test / total_number)
# random_train_fraction = total_test_fraction - (num_fixed_test / total_number)
# fixed_test = np.array(range(num_fixed_test))

# assert total_test_fraction + train_fraction <= 1

# # making the assignments
# all_idxs = range(num_fixed_test, total_number)
# train_idxs, random_test_idxs = train_test_split(all_idxs, test_size=random_test_fraction, train_size=train_fraction, random_state=42)
# test_idxs = np.concatenate((fixed_test, random_test_idxs))


# now doing as boolean arrays
train_binary = np.zeros((total_number), dtype=bool)
train_binary[train_idxs] = True
test_binary = np.zeros((total_number), dtype=bool)
test_binary[test_idxs] = True

# assert no overlap
mistakes = np.sum(np.logical_and(train_binary, test_binary))
print "There are " + str(mistakes) + " mistakes"
assert(mistakes==0)


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

print test_names[:10]
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