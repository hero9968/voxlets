import sys
import os
sys.path.append(os.path.expanduser('~/projects/shape_sharing/src/'))
import real_data_paths as paths

savepath = '/tmp/leaves/'

if not os.path.exists(savepath):
    os.mkdir(savepath)

import cPickle as pickle

with open(paths.RenderedData.voxlet_model_oma_path, 'r') as f:
	model = pickle.load(f)

model.render_leaf_nodes(savepath)