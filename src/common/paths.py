'''
This module is purely here to store the paths to the data associated with all this structured prediction
'''

import os
import socket
import numpy as np

# per-view data paths
host_name = socket.gethostname()
if host_name == 'troll':
    base_path = os.path.expanduser("/mnt/scratch/mfirman/data/")
else:
    base_path = os.path.expanduser("~/projects/shape_sharing/data/")


# paths for the sparse results
results_folder = base_path + "structured/results/"
results_folder_small = base_path + "structured/results_small/"

bigbird_folder = base_path + 'bigbird/'
bigbird_objects = ['coffee_mate_french_vanilla']

data_type = 'bigbird'


if data_type=='bigbird':
    # paths to do with the dataset as a whole
    models_list = base_path + 'bigbird/bb_to_use.txt'
    train_path = base_path + 'bigbird/bb_train.txt'
    test_path = base_path + 'bigbird/bb_test.txt'
    feature_path = base_path + 'bigbird_features/'

    f = open(base_path+'bigbird/poses_to_use.txt', 'r')
    views = [line.strip() for line in f]
    f.close()

    f = open(train_path, 'r')
    train_names = [line.strip() for line in f]
    train_names = [name for name in train_names if name != 'cup_noodles_shrimp_picante' and name != 'paper_plate']
    f.close()
    
    f = open(test_path, 'r')
    test_names = [line.strip() for line in f]
    test_names = [name for name in test_names  if name != 'cup_noodles_shrimp_picante' and name != 'paper_plate']
    f.close()

    model_features = base_path + 'bigbird_features/'

    combined_features_path = base_path + "bigbird_combined/"

    # paths for the random forest models
    model_config = base_path + 'bigbird/models_config.yaml'
    rf_folder_path = base_path + "bigbird_models/rf_models/"
    rf_folder_path_small = base_path + "bigbird_models/rf_models_small/"

    dense_savefolder = base_path + "bigbird_dense/"


    bigbird_training_data_mat = base_path + "voxlets/bigbird/%s.mat"
    bigbird_training_data_mat_tsdf = base_path + "voxlets/bigbird/%s_tsdf.mat"

    voxlet_dict_path = base_path + "voxlets/dict/dict_from_training_images.pkl"
    voxlet_dict_path_tsdf = base_path + "voxlets/dict/dict_from_training_images_tsdf.pkl"
    voxlet_model_path = base_path + "voxlets/dict/forest.pkl"

    # setting some voxlet params here
    # NOTE BE VERY CAREFUL IF EDITING THESE
    one_side_bins = 20
    voxlet_shape = (one_side_bins, 2*one_side_bins, one_side_bins)
    voxlet_size = 0.1/float(one_side_bins)
    voxlet_centre = np.array((0.05, 0.025, 0.05))

elif data_type=='cad':
    # paths to do with the dataset as a whole
    models_list = base_path + 'basis_models/databaseFull/fields/models.txt'
    split_path = base_path + 'structured/split.mat'
    feature_path = base_path + 'structured/features_nopatch/'

    model_features = base_path + 'structured/features/'

    combined_features_path = base_path + 'structured/combined_features/'

    views = list(range(42)) # how many rendered views there are of each object

    # paths for the random forest models
    model_config = base_path + 'models_config.yaml'
    rf_folder_path = base_path + "structured/rf_models/"
    rf_folder_path_small = base_path + "structured/rf_models_small/"

    dense_savefolder = base_path + "structured/dense_predictions/"


# locations of the combined features
combined_test_features = combined_features_path + 'test.pkl'
combined_test_features_small = combined_features_path + 'test_small.pkl'
combined_train_features = combined_features_path + 'train.pkl'
combined_train_features_small = combined_features_path + 'train_small.pkl'

# paths for the dense predictions
#dense_predictions = base_path + 

# create a dictionary of all the model names - for the synthetic models!
f = open(models_list, 'r')
modelname_to_idx = dict()
modelnames = []
for idx, line in enumerate(f):
    modelname = line.strip()
    if modelname == 'cup_noodles_shrimp_picante' or modelname == 'paper_plate':
        continue    
    modelname_to_idx[modelname] = idx
    modelnames.append(modelname)
f.close()
#print modelnames
#print d['12bfa757452ae83d4c5341ee07f41676']


# some helper functions...
def num_files_in_dir(dirname):
    return len([name for name in os.listdir(dirname) if os.path.isfile(os.path.join(dirname, name))])
