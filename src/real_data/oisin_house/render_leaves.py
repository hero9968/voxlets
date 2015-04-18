import sys
import os
import cPickle as pickle

sys.path.append(os.path.expanduser('~/projects/shape_sharing/src/'))

import real_data_paths as paths

savepath = paths.models_folder + '/leaves/'

if not os.path.exists(savepath):
    os.mkdir(savepath)

with open(paths.voxlet_model_oma_path, 'r') as f:
    model = pickle.load(f)

model.render_leaf_nodes(savepath)