import os
import yaml

data_folder = '/Users/Michael/projects/shape_sharing/data/desks/'
raw_data = data_folder + 'oisin_1/data/'
converter_path = '/Users/Michael/projects/InfiniTAM_Alt/convertor/voxels_to_ply.py'

scene_names = [o
          for o in os.listdir(raw_data)
          if os.path.isdir(os.path.join(raw_data,o))]

scene_names = ['saved_aaron']

scenes = [{'folder':raw_data,
           'scene':scene}
           for scene in scene_names]

yaml_train_location = data_folder + 'train_test/train.yaml'
with open(yaml_train_location, 'r') as f:
    train_data = yaml.load(f)

for t in train_data:
    t['folder'] = raw_data

# saving...
models_folder = data_folder + 'models/'

voxlets_dict_data_path = models_folder + 'dictionary/dict_data/'
voxlets_dictionary_path = models_folder + 'dictionary/'
voxlets_data_path = models_folder + 'training_voxlets/'
voxlet_model_oma_path = models_folder + 'models/oma.pkl'
