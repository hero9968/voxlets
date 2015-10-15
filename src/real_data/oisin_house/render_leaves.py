import sys
import os
import cPickle as pickle

sys.path.append(os.path.expanduser('~/projects/shape_sharing/src/'))
from common import rendering
import real_data_paths as paths

# feature = 'cobweb'
voxlet_type = 'short_cobweb_0.002'

savepath = (paths.models_folder % voxlet_type) + '/leaves/'

if not os.path.exists(savepath):
    os.mkdir(savepath)

loadpath = paths.voxlet_model_path % voxlet_type

with open(loadpath, 'r') as f:
    model = pickle.load(f)

print len(model.forest.trees)

height = voxlet_type.split('_')[0]
rendering.render_leaf_medioids(model, savepath, height=height)
