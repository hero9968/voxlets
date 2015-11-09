import os
import yaml
import socket
import system_setup
from copy import deepcopy

host_name = socket.gethostname()
print host_name

if host_name in ['troll', 'biryani', 'dhansak']:
    data_folder = '/home/michael/projects/shape_sharing/data/cleaned_3D/renders_yaml_format/silberman_split/'
else:
    raise Exception('Unknown system')

raw_data = data_folder + '../renders/'

scene_names = [o
          for o in os.listdir(raw_data)
          if os.path.isdir(os.path.join(raw_data,o))]

# scene_names = ['saved_aaron']

scenes = [{'folder':raw_data,
           'scene':scene}
           for scene in scene_names]


yaml_train_location = data_folder + '../splits/train_silberman.yaml'
yaml_test_location = data_folder + '../splits/test_silberman.yaml'

with open(yaml_train_location, 'r') as f:
    all_train_data = yaml.load(f)

with open(yaml_test_location, 'r') as f:
    test_data = yaml.load(f)

test_data = test_data[:175]

if system_setup.small_sample:
    all_train_data = all_train_data[:system_setup.max_sequences]

# saving...
implicit_folder = data_folder + 'implicit/'
implicit_model_dir = implicit_folder + 'models/%s/'
implicit_training_dir = implicit_folder + 'models/%s/training_data/'
implicit_predictions_dir = implicit_folder + 'models/%s/predictions/%s/'

# pre-save the regions to test over...
new_pred_folder = '/media/michael/Seagate/phd_projects/volume_completion_data/data/nyu_cad/predictions/'

evaluation_region_path = new_pred_folder + '/%s/%s/evaluation_region.mat'
