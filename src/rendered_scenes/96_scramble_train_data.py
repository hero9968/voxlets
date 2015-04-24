'''
stupid one-time script to mix up the order of the training data.
This ensures that we get a good mix of scenes in the training, instead of loads
of views of the same scene...
'''
import sys, os
sys.path.append(os.path.expanduser('~/projects/shape_sharing/src/'))
from common import paths

import yaml
import random


with open(paths.RenderedData.yaml_train_location.replace('.yaml', '_backup.yaml'), 'r') as f:
    train_list = yaml.load(f)

with open(paths.RenderedData.yaml_test_location.replace('.yaml', '_backup.yaml'), 'r') as f:
    test_list = yaml.load(f)

print len(train_list), len(test_list)
random.shuffle(train_list)
random.shuffle(test_list)
print len(train_list), len(test_list)

with open(paths.RenderedData.yaml_train_location, 'w') as f:
    yaml.dump(train_list, f)

with open(paths.RenderedData.yaml_test_location, 'w') as f:
    yaml.dump(test_list, f)
