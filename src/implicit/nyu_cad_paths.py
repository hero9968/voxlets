import os
import yaml
import socket
import system_setup
from copy import deepcopy

host_name = socket.gethostname()
print host_name

if host_name in ['troll', 'biryani', 'dhansak']:
    data_folder = '/home/michael/projects/shape_sharing/data/cleaned_3D/renders_yaml_format/'
else:
    raise Exception('Unknown system')

raw_data = data_folder + 'renders/'

scene_names = [o
          for o in os.listdir(raw_data)
          if os.path.isdir(os.path.join(raw_data,o))]

# scene_names = ['saved_aaron']

scenes = [{'folder':raw_data,
           'scene':scene}
           for scene in scene_names]


yaml_train_location = data_folder + 'splits/train.yaml'
yaml_test_location = data_folder + 'splits/test.yaml'

with open(yaml_train_location, 'r') as f:
    all_train_data = yaml.load(f)

with open(yaml_test_location, 'r') as f:
    test_data = yaml.load(f)

test_data = test_data[:system_setup.max_test_sequences]

if system_setup.small_sample:
    all_train_data = all_train_data[:system_setup.max_sequences]

# saving...
implicit_folder = data_folder + 'implicit/'
implicit_model_dir = implicit_folder + 'models/%s/'
implicit_training_dir = implicit_folder + 'models/%s/training_data/'
implicit_predictions_dir = implicit_folder + 'models/%s/predictions/%s/'
