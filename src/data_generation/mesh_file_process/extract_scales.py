'''
idea is to extract the scales from the scene text files and put in a dictionary
note that some models have multiple scales. I naively here just overwrite any 
existing ones with the latest scale
'''
import os, sys
import cPickle as pickle

sys.path.append(os.path.expanduser('~/projects/shape_sharing/src/'))
from common import mesh

data_dir = '/Volumes/HDD/data/others_data/mesh_datasets/databaseFull/'
scene_file = data_dir + 'scenes/scene%05d.txt'

local_mesh_dir = os.path.expanduser('~/projects/shape_sharing/data/meshes2/')
dict_save_path = local_mesh_dir+ 'scales.pkl'

# getting all the scales from the scenes
scale_dict = {}

for i in range(132):

    current_model = None
    current_scale = None

    with open(scene_file % i, 'r') as f:

        for line in f:

            split_line = line.split(' ')

            # seeing if this is a line defining a model
            if split_line[0].strip() == 'newModel' and split_line[1].strip() != '0':
                current_model = split_line[2].strip()

            # seeing if this is a line defining the scale
            if split_line[0].strip() == 'scale' and current_model:

                # get current scale
                current_scale = float(split_line[1].strip())

                # see if already in dict
                #if current_model in scale_dict:
                #    print scale_dict[current_model], current_scale

                # add to dictionary
                scale_dict[current_model] = current_scale

                # reset the counters
                current_model = None
                current_scale = None


# saving dictionary
pickle.dump(scale_dict, open(dict_save_path, 'wb'))
