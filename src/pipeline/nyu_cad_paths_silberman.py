import os
import yaml
import socket
import system_setup
from copy import deepcopy

host_name = socket.gethostname()

data_folder = '/media/michael/Seagate/phd_projects/volume_completion_data/data/nyu_cad_from_SSD/cleaned_3D/renders_yaml_format/'

raw_data = data_folder + 'renders_for_release/'

scene_names = [o
          for o in os.listdir(raw_data)
          if os.path.isdir(os.path.join(raw_data,o))]

scenes = [{'folder':raw_data,
           'scene':scene}
           for scene in scene_names]

# maybe here do ginput or something... with a timeout? what?
yaml_train_location = data_folder + 'splits/train_silberman.yaml'
yaml_test_location = data_folder + 'splits/test_silberman.yaml'

with open(yaml_train_location, 'r') as f:
    all_train_data = yaml.load(f)

with open(yaml_test_location, 'r') as f:
    test_data = yaml.load(f)

# just taking 200 test scenes for time purposes
test_data = test_data[:200]

for idx in range(len(test_data)):
    test_data[idx]['folder'] = data_folder + 'renders_for_release/'

if system_setup.small_sample:
    all_train_data = all_train_data[:200]

print "There are %d train sequences" % len(all_train_data)
print "There are %d test sequences" % len(test_data)

for t in scenes:
    t['folder'] = raw_data

for t in all_train_data:
    t['folder'] = raw_data


# saving...
#                                     vv this is the datatype
models_folder = '../../data_nyu/models/%s/'
training_data_folder = '../../data_nyu/training_voxlets/%s/'

# voxlets_dict_data     _path = models_folder + 'dictionary/dict_data/'
voxlets_dictionary_path = training_data_folder + 'dictionary/'
voxlets_data_path = training_data_folder + 'training_voxlets/'

voxlet_model_path = models_folder + 'model.pkl'

# this is where to save the voxlets used for testing the models
evaluation_data_path = models_folder + 'model_evaluation_voxlets/'

# voxlet_prediction_image_path = base_path + "/voxlets/bigbird/predictions/%s/%s_%s.png"
# new_pred_folder = data_folder_alt + '/predictions'
new_pred_folder = '../../data_nyu/predictions/'
voxlet_prediction_img_path = new_pred_folder + '/%s/%s/%s.png'

evaluation_region_path = new_pred_folder + '/%s/%s/evaluation_region.mat'

# first %s is the test batch category name, second is the sequence name
prediction_folderpath = new_pred_folder + '/%s/%s/pickles/'

scores_path = new_pred_folder + '/%s/%s/scores.yaml'

# final %s is the actual test being done
prediction_path = new_pred_folder + '/%s/%s/%s.pkl'
