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

# maybe here do ginput or something... with a timeout? what?
yaml_train_location = data_folder + 'splits/train_silberman.yaml'
yaml_test_location = data_folder + 'splits/test_silberman.yaml'

with open(yaml_train_location, 'r') as f:
    all_train_data = yaml.load(f)

with open(yaml_test_location, 'r') as f:
    test_data = yaml.load(f)

test_data = test_data#[:175]#[:system_setup.max_test_sequences]
# all_train_data = [test_data[2]]
# test_data = [test_data[2]]
# test_data = [tt for tt in test_data if 'bedroom' in tt['scene']]
# all_train_data = [tt for tt in all_train_data if 'bedroom' in tt['scene']]

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

data_folder_alt = data_folder + "silberman_split/"

# saving...
#                                     vv this is the datatype
models_folder = data_folder_alt + 'models/%s/'

training_data_folder = data_folder_alt + 'training_voxlets/%s/'

# voxlets_dict_data     _path = models_folder + 'dictionary/dict_data/'
voxlets_dictionary_path = training_data_folder + 'dictionary/'
voxlets_data_path = training_data_folder + 'training_voxlets/'

voxlet_model_path = models_folder + 'model.pkl'

# this is where to save the voxlets used for testing the models
evaluation_data_path = models_folder + 'model_evaluation_voxlets/'

# voxlet_prediction_image_path = base_path + "/voxlets/bigbird/predictions/%s/%s_%s.png"
# new_pred_folder = data_folder_alt + '/predictions'
new_pred_folder = '/media/michael/Seagate/phd_projects/volume_completion_data/data/nyu_cad/predictions/'
voxlet_prediction_img_path = new_pred_folder + '/%s/%s/%s.png'

evaluation_region_path = new_pred_folder + '/%s/%s/evaluation_region.mat'

# first %s is the test batch category name, second is the sequence name
prediction_folderpath = new_pred_folder + '/%s/%s/pickles/'

scores_path = new_pred_folder + '/%s/%s/scores.yaml'

# final %s is the actual test being done
prediction_path = new_pred_folder + '/%s/%s/%s.pkl'


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


# defining these here so I can load in zheng results to compare
implicit_folder = data_folder_alt + 'implicit/'
implicit_predictions_dir = implicit_folder + 'models/%s/predictions/%s/'
