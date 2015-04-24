import real_data_paths as paths
import yaml
import random
import os
from copy import deepcopy
random.seed(10)

# warning - make sure training on the dark scenes and testing on bright ones, for visualisation purposes...
def add_to_scene(t):

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
    return temp


for (folder, saveloc) in zip(['data1/', 'data2/'],
    [paths.yaml_train_location, paths.yaml_train_location2]):


    raw_data = paths.data_folder + folder

    scene_names = [o
              for o in os.listdir(raw_data)
              if os.path.isdir(os.path.join(raw_data,o))]

    scenes = [{'folder':raw_data,
               'scene':scene}
               for scene in scene_names]

    scenes = [add_to_scene(s) for s in scenes]
    scenes = [t for temp in scenes for t in temp]

    random.shuffle(scenes)

    with open(saveloc, 'w') as f:
        yaml.dump(scenes, f)
