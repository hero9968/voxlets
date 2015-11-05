# create new split from the real silberman split

import scipy.io
import os, sys
import numpy
import yaml
import random

base_dir = '/home/michael/projects/shape_sharing/data/cleaned_3D/'
new_dir = base_dir + 'renders_yaml_format/renders/'
splits_dir = base_dir + 'renders_yaml_format/splits/'

# create the dictionary
all_seq = {}
for fname in os.listdir(new_dir):
    number = int(fname.split('_')[0])
    all_seq[number] = {
        'frames': [0],
        'name': fname,
        'scene': fname,
        'folder': new_dir
    }

D = scipy.io.loadmat(splits_dir + "splits.mat")
train_seq = [all_seq[xx] for xx in D['trainNdxs'].ravel()]
test_seq = [all_seq[xx] for xx in D['testNdxs'].ravel()]

print "Train seq is len ", len(train_seq)
print "Test seq is len ", len(test_seq)

random.seed(10)
random.shuffle(test_seq, )

# save train and test sequence to file
with open(splits_dir + 'test_silberman.yaml', 'w') as f:
    yaml.dump(test_seq, f, default_flow_style=False)

with open(splits_dir + 'train_silberman.yaml', 'w') as f:
    yaml.dump(train_seq, f, default_flow_style=False)
