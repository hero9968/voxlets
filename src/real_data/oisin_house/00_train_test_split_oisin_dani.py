import real_data_paths as paths
import yaml
import random

random.seed(10)

for folder, saveloc in zip(['data1/', 'data/'],
    [paths.yaml_train_location, paths.yaml_test_location])

    raw_data = paths.data_folder + folder

    scene_names = [o
              for o in os.listdir(raw_data)
              if os.path.isdir(os.path.join(raw_data,o))]

    scenes = [{'folder':raw_data,
               'scene':scene}
               for scene in scene_names]

    random.shuffle(scenes)

    with open(saveloc, 'w') as f:
        yaml.dump(scenes, f)
