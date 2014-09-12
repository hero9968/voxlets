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
    base_path = os.path.expanduser("~/projects/shape_sharing/data/3D/basis_models/")

model_features = base_path + 'structured/features/'

# paths to do with the dataset as a whole
models_list = base_path + 'databaseFull/fields/models.txt'
split_path = base_path + 'structured/split.mat'

# locations of the combined features
combined_features_path = base_path + 'structured/combined_features/'
combined_test_features = combined_features_path + 'test.pkl'
combined_test_features_small = combined_features_path + 'test_small.pkl'
combined_train_features = combined_features_path + 'train.pkl'
combined_train_features_small = combined_features_path + 'train_small.pkl'

# paths for the random forest models
model_config = base_path + 'models_config.yaml'
rf_folder_path = base_path + "rf_models/"
rf_folder_path_small = base_path + "rf_models_small/"

# paths for the sparse results
results_folder = "./data/results/"
results_folder_small = "./data/results_small/"

# paths for the dense predictions
#dense_predictions = base_path + 