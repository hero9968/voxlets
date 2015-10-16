import os, sys
import numpy
import yaml

base_dir = '/home/michael/projects/shape_sharing/data/cleaned_3D/'
new_dir = base_dir + 'renders_yaml_format/renders/'
splits_dir = base_dir + 'renders_yaml_format/splits/'


max_sequences = 2

# create the dictionary
train_seq = []
for fname in os.listdir(new_dir):
    train_seq.append(
    {
        'frames': [0],
        'name': fname,
        'scene': fname,
        'folder': new_dir
    })

train_seq = train_seq[:max_sequences]
test_seq = train_seq

# save train and test sequence to file
for savename, seq in [('train.yaml', train_seq), ('test.yaml', test_seq)]:
    with open(splits_dir + savename, 'w') as f:
        yaml.dump(seq, f, default_flow_style=False)
