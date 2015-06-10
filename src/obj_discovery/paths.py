'''
set up the paths for the system
'''

import os
import yaml
import socket
import system_setup
from copy import deepcopy

host_name = socket.gethostname()
print host_name

if host_name == 'troll' or host_name == 'biryani':
    data_folder = '/media/ssd/data/oisin_house/'
    converter_path = ''
else:
    data_folder = '/Users/Michael/projects/shape_sharing/data/oisin_house/'
    converter_path = '/Users/Michael/projects/InfiniTAM_Alt/convertor/voxels_to_ply.py'

raw_data = data_folder + 'data2/'

scene_names = [o
          for o in os.listdir(raw_data)
          if os.path.isdir(os.path.join(raw_data,o))]

# scene_names = ['saved_aaron']

scenes = [{'folder':raw_data,
           'scene':scene}
           for scene in scene_names]

yaml_train_location = data_folder + 'train_test/train.yaml'
yaml_train_location2 = data_folder + 'train_test/train2.yaml'
yaml_test_location = data_folder + 'train_test/test.yaml'

with open(yaml_train_location, 'r') as f:
    temp_train_data = yaml.load(f)

with open(yaml_train_location2, 'r') as f:
    temp_train_data2 = yaml.load(f)

all_train_data = temp_train_data + temp_train_data2

with open(yaml_test_location, 'r') as f:
    test_data = yaml.load(f)

# The folders and filenames for saving
obj_disc_dir = data_folder + 'obj_discovery/'

labels_dir = obj_disc_dir + '/labelling/labels/'

segmented_dir = obj_disc_dir + 'segmented/'
segmented_path = segmented_dir + '%s.mat'

features_dir = obj_disc_dir + 'features/'
features_path = features_dir + '%s.mat'

models_dir = obj_disc_dir + 'models/'