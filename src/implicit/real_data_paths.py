import os
import yaml
import socket
import system_setup
from copy import deepcopy

host_name = socket.gethostname()
print host_name

if host_name == 'troll' or host_name == 'biryani':
    data_folder = '/media/michael/Seagate/phd_projects/volume_completion_data/data/oisin_house/'
    converter_path = ''
else:
    data_folder = '/media/michael/Seagate/phd_projects/volume_completion_data/data/oisin_house/'
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

test_data = test_data[:system_setup.max_test_sequences]

if system_setup.small_sample:
    all_train_data = all_train_data[:system_setup.max_sequences_small]
else:
    all_train_data = all_train_data[:system_setup.max_sequences]

for idx, _ in enumerate(all_train_data):
    ending = all_train_data[idx]['folder'].split('/')[-2]
    all_train_data[idx]['folder'] = data_folder + ending + '/'

for idx, _ in enumerate(test_data):
    ending = test_data[idx]['folder'].split('/')[-2]
    test_data[idx]['folder'] = data_folder + ending + '/'

sequences = []
for t in scenes:
    t['folder'] = raw_data
    fpath = t['folder'] + t['scene'] + '/test_frame.txt'

    with open(fpath, 'r') as f:
        frames = [int(l) for l in f]

    temp = []
    for fr in frames:
        this_t = deepcopy(t)
        this_t['frames'] = [fr]
        this_t['name'] = this_t['scene'] + '_' + str(this_t['frames'])
        temp.append(this_t)

    sequences.append(temp)

# saving...
implicit_folder = data_folder + 'implicit/'
implicit_model_dir = implicit_folder + 'models/%s/'
implicit_training_dir = implicit_folder + 'models/%s/training_data/'
implicit_predictions_dir = implicit_folder + 'models/%s/predictions/%s/'
