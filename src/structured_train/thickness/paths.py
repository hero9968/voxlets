'''
This module is purely here to store the paths to the data associated with all this structured prediction
'''

import os
import socket

# per-view data paths
host_name = socket.gethostname()
if host_name == 'troll':
    base_path = os.path.expanduser("/mnt/scratch/mfirman/data/")
else:
    base_path = os.path.expanduser("~/projects/shape_sharing/data/")

model_features = base_path + 'structured/features/'



# locations of the combined features
combined_features_path = base_path + 'structured/combined_features/'
combined_test_features = combined_features_path + 'test.pkl'
combined_test_features_small = combined_features_path + 'test_small.pkl'
combined_train_features = combined_features_path + 'train.pkl'
combined_train_features_small = combined_features_path + 'train_small.pkl'

# paths for the random forest models
model_config = base_path + 'models_config.yaml'
rf_folder_path = base_path + "structured/rf_models/"
rf_folder_path_small = base_path + "structured/rf_models_small/"

# paths for the sparse results
results_folder = base_path + "structured/results/"
results_folder_small = base_path + "structured/results_small/"

bigbird_folder = base_path + 'bigbird/'
bigbird_objects = ['coffee_mate_french_vanilla']

data_type = 'bigbird'


if data_type=='bigbird':
    # paths to do with the dataset as a whole
    models_list = base_path + 'bigbird/bb_to_use.txt'
    split_path = base_path + 'bigbird/split.mat'
    feature_path = base_path + 'bigbird_features/'

    f = open(base_path+'bigbird/poses_to_use.txt', 'r')
    views = [line.strip() for line in f]

elif data_type=='cad':
    # paths to do with the dataset as a whole
    models_list = base_path + 'basis_models/databaseFull/fields/models.txt'
    split_path = base_path + 'structured/split.mat'
    feature_path = base_path + 'structured/features_nopatch/'

    views = list(range(42)) # how many rendered views there are of each object


# paths for the dense predictions
#dense_predictions = base_path + 

# create a dictionary of all the model names - for the synthetic models!
f = open(models_list, 'r')
modelname_to_idx = dict()
modelnames = []
for idx, line in enumerate(f):
    modelname = line.strip()
    modelname_to_idx[modelname] = idx
    modelnames.append(modelname)

#print modelnames
#print d['12bfa757452ae83d4c5341ee07f41676']
