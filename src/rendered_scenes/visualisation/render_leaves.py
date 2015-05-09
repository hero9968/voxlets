import sys
import os
sys.path.append(os.path.expanduser('~/projects/shape_sharing/src/'))
from common import paths, voxlets

savepath = '/tmp/leaves/'

if not os.path.exists(savepath):
    os.mkdir(savepath)

import cPickle as pickle

print 'Warning - rendering with custom path'
with open('/media/ssd/data/oisin_house/models_full_split_floating_different_split/models/oma_cobweb.pkl', 'r') as f:
	model = pickle.load(f)

voxlets.render_leaf_nodes(model, savepath)