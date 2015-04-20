import os
import yaml
import socket

host_name = socket.gethostname()
print host_name

if host_name == 'troll' or host_name == 'biryani':
    data_folder = '/media/ssd/data/oisin_house/'
    converter_path = ''
else:
    data_folder = '/Users/Michael/projects/shape_sharing/data/oisin_house/'
    converter_path = '/Users/Michael/projects/InfiniTAM_Alt/convertor/voxels_to_ply.py'

raw_data = data_folder + 'data1/'

scene_names = [o
          for o in os.listdir(raw_data)
          if os.path.isdir(os.path.join(raw_data,o))]

# scene_names = ['saved_aaron']

scenes = [{'folder':raw_data,
           'scene':scene}
           for scene in scene_names]

yaml_train_location = data_folder + 'train_test/train.yaml'
yaml_test_location = data_folder + 'train_test/test.yaml'
with open(yaml_train_location, 'r') as f:
    train_data = yaml.load(f)

with open(yaml_test_location, 'r') as f:
    test_data = yaml.load(f)


from copy import deepcopy

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
models_folder = data_folder + 'models_full_split/'

voxlets_dict_data_path = models_folder + 'dictionary/dict_data/'
voxlets_dictionary_path = models_folder + 'dictionary/'
voxlets_data_path = models_folder + 'training_voxlets/'
voxlet_model_oma_path = models_folder + 'models/oma.pkl'

# voxlet_prediction_image_path = base_path + "/voxlets/bigbird/predictions/%s/%s_%s.png"
voxlet_prediction_img_path = data_folder + '/predictions/%s/%s/%s.png'
voxlet_prediction_folderpath = data_folder + '/predictions/%s/%s/'


def new_dropbox_dir():
    '''
    creates a new dropbox directory for results
    '''
    base_path = \
        os.path.expanduser('~/Dropbox/PhD/Projects/Shape_sharing_data/oisin_house_predictions/res_%04d/')
    count = 0
    while os.path.exists(base_path % count):
        count += 1
    os.mkdir(base_path % count)
    assert os.path.exists(base_path % count)
    return base_path % count
