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

print "There are %d train sequences" % len(all_train_data)
print "There are %d test sequences" % len(test_data)

for t in scenes:
    t['folder'] = raw_data

for t in all_train_data:
    t['folder'] = raw_data

# for t in all_train_data:
#     t['folder'] = raw_data

# saving...
#                                     vv this is the datatype
models_folder = data_folder + 'models/%s/'

training_data_folder = data_folder + 'training_voxlets/%s/'

# voxlets_dict_data     _path = models_folder + 'dictionary/dict_data/'
voxlets_dictionary_path = training_data_folder + 'dictionary/'
voxlets_data_path = training_data_folder + 'training_voxlets/'

voxlet_model_path = models_folder + 'model.pkl'

# this is where to save the voxlets used for testing the models
evaluation_data_path = models_folder + 'model_evaluation_voxlets/'

# voxlet_prediction_image_path = base_path + "/voxlets/bigbird/predictions/%s/%s_%s.png"
voxlet_prediction_img_path = data_folder + '/predictions/%s/%s/%s.png'

# first %s is the test batch category name, second is the sequence name
prediction_folderpath = data_folder + '/predictions/%s/%s/pickles/'

scores_path = data_folder + '/predictions/%s/%s/scores.yaml'

# final %s is the actual test being done
prediction_path = data_folder + '/predictions/%s/%s/%s.pkl'


def new_dropbox_dir():
    '''
    creates a new dropbox directory for results
    '''
    base_path = \
        os.path.expanduser('~/Dropbox/PhD/Projects/Shape_sharing_data/synthetic_predictions/res_%04d/')
    count = 0
    while os.path.exists(base_path % count):
        count += 1
    os.mkdir(base_path % count)
    assert os.path.exists(base_path % count)
    return base_path % count
