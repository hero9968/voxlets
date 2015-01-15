'''
this script will form the train test split between
the intrinsic scenes
'''
import numpy as np
import sys, os
import yaml
from sklearn.cross_validation import train_test_split
import random
import string

sys.path.append(os.path.expanduser('~/projects/shape_sharing/src/'))
from common import paths

# parameters
num_sequences_per_scene = 10
num_scenes_to_use = 1
num_frames_per_sequence = 5
train_test_overlap = True
if train_test_overlap:
  num_train = 8  # integers as there is an overlap
  num_test = 1
  assert(num_train + num_test <= num_sequences_per_scene * num_scenes_to_use)
else:
  train_fraction = 0.6  # direct split, so express as fraction



def load_scene(scene_name):
    with open(paths.scenes_location + scene_name + '/poses.yaml') as f:
        return (scene_name, yaml.load(f))

scene_names_to_use = paths.rendered_primitive_scenes[:num_scenes_to_use]
print "Using a total of %d scenes" % len(scene_names_to_use)

all_scenes = [load_scene(scene_name)
              for scene_name in scene_names_to_use]


def random_sequence(length_to_draw_from, number_to_draw):
    '''
    returns a list of number_to_draw consequtive numbers
    from range(length_to_draw_from)
    '''
    start = np.random.randint(0, length_to_draw_from - number_to_draw)
    end = start + number_to_draw
    return range(start, end)


def random_string(length):
    '''
    returns a string of random characters, of length 'length'
    '''
    return ''.join(random.choice(string.ascii_lowercase + string.digits)
                   for _ in range(length))


# choosing which frames to use from each video
# loop over each scene in total
    # choose which frames from this scene are to be used
sequences = {}
for idx, (scene_name, scene) in enumerate(all_scenes):
    sequences[scene_name] = [random_sequence(len(scene),
                             num_frames_per_sequence)
                             for i in range(num_sequences_per_scene)]

print "There are %d sequences" % len(sequences['YP8G55G2GZ'])

# making split at a scene level
if train_test_overlap:
    # here there is an overlap, so just take some random sequences from
    # some random sceens
    #num_train = np.ceil(train_fraction * len(scene_names_to_use))
    #num_test = np.ceil((1 - train_fraction) * len(scene_names_to_use))

    all_sequences = [(scene_name, seq)
                     for seq in sequences[scene_name]
                     for scene_name in scene_names_to_use]

    train_sequence, test_sequence = train_test_split(
        all_sequences, train_size = num_train, test_size = num_test)
    train_list = [dict(name=random_string(16), scene=seq[0], frames=seq[1])
                  for seq in train_sequence]
    test_list = [dict(name=random_string(16), scene=seq[0], frames=seq[1])
                  for seq in test_sequence]

else:
    # here we are doing a full split between the sides
    train_scenes, test_scenes = train_test_split(
        scene_names_to_use, train_size=train_fraction)

    print "After split: %d training scenes, %d test scenes" % \
          (len(train_scenes), len(test_scenes))

    # forming the full training and test lists
    test_list = [dict(name=random_string(16), scene=scene_name, frames=s)
                 for scene_name in test_scenes
                 for s in sequences[scene_name]]

    train_list = [dict(name=random_string(16), scene=scene_name, frames=s)
                  for scene_name in train_scenes
                  for s in sequences[scene_name]]

# writing all data to a yaml file
with open(paths.yaml_train_location, 'w') as f:
    yaml.dump(train_list, f)

with open(paths.yaml_test_location, 'w') as f:
    yaml.dump(test_list, f)
