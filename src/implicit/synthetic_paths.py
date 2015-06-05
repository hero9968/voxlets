'''
in hindsight I should have all the paths defined in a yaml file
this could then be given as an input
'''
import os
import yaml
import socket
import system_setup
from copy import deepcopy

host_name = socket.gethostname()
print host_name

if host_name == 'troll' or host_name == 'biryani':
    data_folder = '/media/ssd/data/rendered_arrangements/'
else:
    data_folder = '/Users/Michael/projects/shape_sharing/data/rendered_scenes'

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

for train_datum in all_train_data:
    train_datum['folder'] = raw_data

for test_datum in test_data:
    test_datum['folder'] = raw_data


# sequences = []
# for t in scenes:
#     t['folder'] = raw_data
#     fpath = t['folder'] + t['scene'] + '/test_frame.txt'

#     with open(fpath, 'r') as f:
#         frames = [int(l) for l in f]

#     temp = []
#     for fr in frames:
#         this_t = deepcopy(t)
#         this_t['frames'] = [fr]
#         this_t['name'] = this_t['scene'] + '_' + str(this_t['frames'])
#         temp.append(this_t)

#     sequences.append(temp)

# saving...
implicit_folder = data_folder + 'implicit/'
implicit_model_dir = implicit_folder + 'models/%s/'
implicit_training_dir = implicit_folder + 'models/%s/training_data/'
implicit_predictions_dir = implicit_folder + 'models/%s/predictions/%s/'


