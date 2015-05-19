import sys
import os
import cPickle as pickle

sys.path.append(os.path.expanduser('~/projects/shape_sharing/src/'))
from common import rendering
import real_data_paths as paths

feature = 'cobweb'
voxlet_type = 'short'

savepath = (paths.models_folder % voxlet_type) + '/leaves/'

if not os.path.exists(savepath):
    os.mkdir(savepath)

loadpath = paths.voxlet_model_path % (voxlet_type, feature)

with open(loadpath, 'r') as f:
    model = pickle.load(f)

rendering.render_leaf_medioids(model, savepath, height=voxlet_type)