import sys
import os
sys.path.append(os.path.expanduser('~/projects/shape_sharing/src/'))
from common import paths
from common import parameters
from common import voxel_data

import scipy.io
import numpy as np
import cPickle as pickle
import shutil

from skimage import measure
import subprocess as sp

sys.path.append(os.path.expanduser(
    '~/projects/shape_sharing/src/rendered_scenes/visualisation'))
import voxel_utils as vu

shoeboxes = []
features = []

def render_single_voxlet(V, savepath, level=0):

    # renders a voxlet using the .blend file...
    temp = V.copy()
    # temp = np.pad(temp, ((1, 1), (1, 1), (1, 1)), 'constant',

    #V[:, :, -2:] = parameters.RenderedVoxelGrid.mu
    verts, faces = measure.marching_cubes(V, level)

    verts *= parameters.Voxlet.size
    verts *= 10.0  # so its a reasonable scale for blender
    print verts.min(axis=0), verts.max(axis=0)
    vu.write_obj(verts, faces, '/tmp/temp_voxlet.obj')
    sp.call([paths.blender_path,
             paths.RenderedData.voxlet_render_blend,
             "-b", "-P",
             paths.RenderedData.voxlet_render_script])#,
             #stdout=open(os.devnull, 'w'),
             #close_fds=True)

    #now copy file from /tmp/.png to the savepath...
    print "Moving render to " + savepath
    shutil.move('/tmp/temp_voxlet.png', savepath)

features = []
pca_representation = []

for count, sequence in enumerate(paths.RenderedData.train_sequence()):

    # loading the data
    loadpath = paths.RenderedData.voxlets_data_path + \
        sequence['name'] + '.mat'
    print "Loading from " + loadpath

    D = scipy.io.loadmat(loadpath)
    features.append(D['features'])
    pca_representation.append(D['shoeboxes'])

    if count > parameters.max_sequences:
        print "SMALL SAMPLE: Stopping"
        break



np_all_sboxes = np.concatenate(pca_representation, axis=0)
print np_all_sboxes.shape
np.random.shuffle(np_all_sboxes)

pca_savepath = paths.RenderedData.voxlets_dictionary_path + 'shoeboxes' + '_pca.pkl'

with open(pca_savepath, 'rb') as f:
    pca = pickle.load(f)

for i in range(100):
    V_temp = np_all_sboxes[i, :]
    #V_pca = pca.transform(V_temp)
    V = pca.inverse_transform(V_temp)
    render_single_voxlet(V.reshape(parameters.Voxlet.shape),
        '/tmp/training/pca_render_' + str(i) + '.png')
