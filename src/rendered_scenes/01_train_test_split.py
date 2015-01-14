'''
this script will form the train test split between
the intrinsic scenes
'''
import numpy as np
import sys, os
import yaml
from sklearn.cross_validation import train_test_split

sys.path.append(os.path.expanduser('~/projects/shape_sharing/src/'))
from common import paths

# parameters
num_frames_per_sequence = 5
num_sequences_per_scene = 10
train_fraction = 0.6
train_test_overlap = False

def load_scene(scene_name):
    with open(paths.scenes_location + scene_name + '/poses.yaml') as f:
        return (scene_name, yaml.load(f))

all_scenes = [load_scene(scene_name) for scene_name in paths.rendered_primitive_scenes]


def random_sequence(length_to_draw_from, number_to_draw):
    '''
    returns a list of number_to_draw consequtive numbers
    from range(length_to_draw_from)
    '''
    start = np.random.randint(0, length_to_draw_from - number_to_draw)
    print start
    end = start + number_to_draw
    return range(start, end)


# choosing which frames to use from each video
# loop over each scene in total
    # choose which frames from this scene are to be used
sequences = {}
for idx, (scene_name, scene) in enumerate(all_scenes):
    sequences[scene_name] = [random_sequence(len(scene), num_frames_per_sequence)
                            for i in range(num_sequences_per_scene)]


# making split at a scene level
if train_test_overlap == False:
    train_scenes, test_scenes = train_test_split(
        paths.rendered_primitive_scenes, train_size=train_fraction)


# forming the full training and test lists
test_list = [dict(scene=name, frames=s) for name in test_scenes for s in sequences[name]]
train_list = [dict(scene=name, frames=s) for name in train_scenes for s in sequences[name]]


# writing all data to a yaml file
with open(paths.train_location, 'w') as f:
    yaml.dump(train_list, f)

with open(paths.test_location, 'w') as f:
    yaml.dump(test_list, f)
