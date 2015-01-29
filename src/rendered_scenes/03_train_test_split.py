'''
this script will form the train test split between
the intrinsic scenes
'''
import numpy as np
import sys
import os
import yaml
from sklearn.cross_validation import train_test_split
import random
import string

sys.path.append(os.path.expanduser('~/projects/shape_sharing/src/'))
from common import paths
from common import parameters

random.seed(10)


def load_scene(scene_name):
    scene_yaml_file = paths.RenderedData.video_yaml(scene_name)
    with open(scene_yaml_file) as f:
        return (scene_name, yaml.load(f))

scene_names_to_use = paths.RenderedData.get_scene_list()[
    :parameters.RenderData.train_test_max_scenes]
print "Using a total of %d scenes" % len(scene_names_to_use)

all_scenes = [load_scene(scene_name) for scene_name in scene_names_to_use]


def random_sequence(length_to_draw_from, number_to_draw):
    '''
    returns a list of number_to_draw consecutive numbers
    from range(length_to_draw_from)
    '''
    start = np.random.randint(0, length_to_draw_from - number_to_draw)
    end = start + number_to_draw
    return range(start, end)


def random_seq_name(length):
    '''
    returns a string of random characters, of length 'length'
    '''
    return ''.join(random.choice(string.ascii_lowercase + string.digits)
                   for _ in range(length)) + '_SEQ'


# choosing which frames to use from each video
# loop over each scene in total
    # choose which frames from this scene are to be used
sequences = {}
for idx, (scene_name, scene) in enumerate(all_scenes):
    sequences[scene_name] = [
        random_sequence(len(scene), parameters.RenderData.frames_per_sequence)
        for i in range(parameters.RenderData.sequences_per_scene)]

print "There are %d sequences" % len(sequences)

# here we are doing a full split between the sides
train_scenes, test_scenes = train_test_split(
    scene_names_to_use, train_size=parameters.RenderData.train_fraction)

print "After split: %d training scenes, %d test scenes" % \
      (len(train_scenes), len(test_scenes))

# forming the full training and test lists
test_list = [dict(name=random_seq_name(16), scene=scene_name, frames=s)
             for scene_name in test_scenes
             for s in sequences[scene_name]]

train_list = [dict(name=random_seq_name(16), scene=scene_name, frames=s)
              for scene_name in train_scenes
              for s in sequences[scene_name]]

# checking if output folder exists
split_save_dir = os.path.dirname(paths.RenderedData.yaml_train_location)
if not os.path.exists(split_save_dir):
    os.makedirs(split_save_dir)


# writing all data to a yaml file
with open(paths.RenderedData.yaml_train_location, 'w') as f:
    yaml.dump(train_list, f)

with open(paths.RenderedData.yaml_test_location, 'w') as f:
    yaml.dump(test_list, f)
