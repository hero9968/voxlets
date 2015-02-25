import sys
import os
sys.path.append(os.path.expanduser('~/projects/shape_sharing/src/'))
from common import paths


import cPickle as pickle

with open(paths.RenderedData.voxlet_model_oma_path, 'r') as f:
	model = pickle.load(f)

model.render_leaf_nodes('/tmp/leaves/')