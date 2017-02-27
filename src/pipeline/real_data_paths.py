import os
import yaml
import system_setup
from copy import deepcopy

# data_folder = '/media/michael/Seagate/phd_projects/volume_completion_data/data/oisin_house/'
data_folder = '/home/michael/Dropbox/Public/for_release/'

raw_data = data_folder + 'fold_2/'

scene_names = [o
          for o in os.listdir(raw_data)
          if os.path.isdir(os.path.join(raw_data,o))]

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
    all_train_data = all_train_data[:system_setup.max_sequences]

# fix the paths...
for item in all_train_data:
    item['folder'] = data_folder + item['folder'].split('/')[-2] + '/'
    item['folder'] = item['folder'].replace('data2', 'fold_2')
    item['folder'] = item['folder'].replace('data', 'fold_0')
    item['folder'] = item['folder'].replace('data1', 'fold_1')

for item in test_data:
    item['folder'] = data_folder + item['folder'].split('/')[-2] + '/'
    item['folder'] = item['folder'].replace('data2', 'fold_2')
    item['folder'] = item['folder'].replace('data', 'fold_0')
    item['folder'] = item['folder'].replace('data1', 'fold_1')

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

#                                     vv this is the datatype
# models_folder = data_folder + 'models/%s/'
#
# training_data_folder = data_folder + 'training_voxlets/%s/'
models_folder = '../../data/models/%s/'
training_data_folder = '../../data/training_voxlets/%s/'

voxlets_dictionary_path = training_data_folder + 'dictionary/'
voxlets_data_path = training_data_folder + 'training_voxlets/'

voxlet_model_path = models_folder + 'model.pkl'

# this is where to save the voxlets used for testing the models
evaluation_data_path = models_folder + 'model_evaluation_voxlets/'
#
# # voxlet_prediction_image_path = base_path + "/voxlets/bigbird/predictions/%s/%s_%s.png"
# voxlet_prediction_img_path = data_folder + '/predictions/%s/%s/%s.png'
voxlet_prediction_img_path = '../../data/predictions/%s/%s/%s.png'
#
# evaluation_region_path = data_folder + '/predictions/%s/%s/evaluation_region.mat'
#
# # first %s is the test batch category name, second is the sequence name
# prediction_folderpath = data_folder + '/predictions/%s/%s/pickles/'
prediction_folderpath = '../../data/predictions/%s/%s/pickles/'
#
scores_path = data_folder + '/predictions/%s/%s/scores.yaml'
#
# # final %s is the actual test being done
# prediction_path = data_folder + '/predictions/%s/%s/%s.pkl'
#
# # final %s is the actual test being done
# kinfu_prediction_img_path = data_folder + '/kinfu_predictions/%s/%s/%s.png'
