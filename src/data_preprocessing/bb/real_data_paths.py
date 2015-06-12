import os
import yaml
import socket

host_name = socket.gethostname()
print host_name

data_folder = '/media/ssd/data/bb/'

train_location = '/media/ssd/data/bigbird/bb_train.txt'
test_location = '/media/ssd/data/bigbird/bb_test.txt'
poses_to_use = '/media/ssd/data/bigbird/poses_to_use.txt'

with open(train_location, 'r') as f:
    train_objects = [l.strip() for l in f]

with open(test_location, 'r') as f:
    test_objects = [l.strip() for l in f]

with open(poses_to_use, 'r') as f:
    poses = [l.strip() for l in f]

# now creating some sequences...
train_data = []
for train_object in train_objects:
    for pose in poses[::3]:
        D= {}
        D['name'] = train_object + '_' + pose
        D['scene'] = train_object
        D['pose_id'] = pose
        D['frames'] = 'ss'
        D['folder'] = '/media/ssd/data/bigbird_cropped/'
        train_data.append(D)

# sparse_train_data = []
# for train_object in train_objects:
#     for pose in poses[::5]:
#         D= {}
#         D['name'] = train_object + '_' + pose
#         D['scene'] = train_object
#         D['pose_id'] = pose
#         D['frames'] = 'ss'
#         D['folder'] = '/media/ssd/data/bigbird_cropped/'
#         sparse_train_data.append(D)


from copy import deepcopy


# saving...
models_folder = data_folder + 'models/'

voxlets_dict_data_path = models_folder + 'dictionary/dict_data/'
voxlets_dictionary_path = models_folder + 'dictionary/'
voxlets_data_path = models_folder + 'training_voxlets/'
voxlet_model_oma_path = models_folder + 'models/oma.pkl'

# voxlet_prediction_image_path = base_path + "/voxlets/bigbird/predictions/%s/%s_%s.png"
voxlet_prediction_img_path = data_folder + '/predictions/%s/%s/%s.png'
voxlet_prediction_folderpath = data_folder + '/predictions/%s'
voxlet_prediction_savepath = data_folder + '/predictions/%s/%s'


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
