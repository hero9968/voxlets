import os
import yaml
import socket

host_name = socket.gethostname()
print host_name

if host_name == 'troll' or host_name == 'biryani':
    data_folder = '/media/ssd/data/nyu/'
    converter_path = ''
else:
    pass
    # data_folder = '/Users/Michael/projects/shape_sharing/data/oisin_house/'
    # converter_path = '/Users/Michael/projects/InfiniTAM_Alt/convertor/voxels_to_ply.py'

raw_data = data_folder + 'data/'

scene_names = [o
          for o in os.listdir(raw_data)
          if os.path.isdir(os.path.join(raw_data,o))]

# scene_names = ['saved_aaron']

scenes = [{'folder':raw_data,
           'scene':scene}
           for scene in scene_names]

test_data = scenes
for t in test_data:
    t['folder'] = raw_data
    t['frames'] = [0]
    t['name'] = t['scene'] + '_[0]'


# saving...
models_folder = data_folder + 'models_full_split/'

voxlet_model_oma_path = '/media/ssd/data/oisin_house/models_full_split_not_tall/models/oma.pkl'

# voxlet_prediction_image_path = base_path + "/voxlets/bigbird/predictions/%s/%s_%s.png"
voxlet_prediction_img_path = data_folder + '/predictions/%s/%s.png'
voxlet_prediction_folderpath = data_folder + '/predictions/%s/'