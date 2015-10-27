import os, sys
import numpy
import yaml

base_dir = '/home/michael/projects/shape_sharing/data/cleaned_3D/'
new_dir = base_dir + 'renders_yaml_format/renders/'
splits_dir = base_dir + 'renders_yaml_format/splits/'

from sklearn.cross_validation import train_test_split

# create the dictionary
all_seq = []
for fname in os.listdir(new_dir):
    all_seq.append(
    {
        'frames': [0],
        'name': fname,
        'scene': fname,
        'folder': new_dir
    })

train_seq, test_seq = train_test_split(all_seq, test_size=0.25)

print "Train seq is len ", len(train_seq)
print "Test seq is len ", len(test_seq)

# save train and test sequence to file
with open(splits_dir + 'test.yaml', 'w') as f:
    yaml.dump(test_seq, f, default_flow_style=False)

with open(splits_dir + 'train.yaml', 'w') as f:
    yaml.dump(train_seq, f, default_flow_style=False)
