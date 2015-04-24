import numpy as np
import cPickle as pickle
import sys
import os
sys.path.append(os.path.expanduser("~/projects/shape_sharing/src/"))
from time import time
import paths
import parameters

from common import voxlets
from common import scene

import sklearn.metrics

with open(paths.RenderedData.voxlets_path + '/models/oma.pkl', 'rb') as f:
    model_without_implicit = pickle.load(f)

print model_without_implicit.training_Y.shape
print model_without_implicit.training_Y.dtype
sadsd

# Path where any renders will be saved to
gen_renderpath = '/media/ssd/data/rendered_arrangements/voxlets/pca_tests/renders/%s/%05d.png'
gt_renderpath = '/media/ssd/data/rendered_arrangements/voxlets/pca_tests/renders/%s/%s.png'
folderpath = '/media/ssd/data/rendered_arrangements/voxlets/pca_tests/renders/%s/'

subsample_length = 2500


# furst load in all the diff pca models
pca = {}
for num_components in [5, 50, 100, 200, 400]:

    basepath = '/media/ssd/data/rendered_arrangements/voxlets/pca_tests/pca_%04d_comp_%06d_samples.pkl'
    savepath = basepath % (num_components, subsample_length)

    print "Num components", num_components
    pca[num_components] = pickle.load(open(savepath, 'r'))
    print pca[num_components].n_components

# now loop per scene
for sequence in paths.RenderedData.test_sequence()[304:]:
    if not os.path.exists(folderpath % sequence['name']):
        os.makedirs(folderpath % sequence['name'])

    sc = scene.Scene(parameters.RenderedVoxelGrid.mu,
        model_without_implicit.voxlet_params)
    sc.load_sequence(sequence, frame_nos=0, segment_with_gt=True,
        save_grids=False, load_implicit=False)

    print "-> Reconstructing with oma forest"
    rec = voxlets.Reconstructer(
        reconstruction_type='kmeans_on_pca', combine_type='modal_vote')
    rec.set_scene(sc)
    rec.sample_points(parameters.VoxletPrediction.number_samples,
                      parameters.VoxletPrediction.sampling_grid_size)

    for num_components in [5, 50, 100, 200, 400]:

        model_without_implicit.pca = pca[num_components]

        rec.set_model(model_without_implicit)
        print "Working with %d comp " % num_components
        print rec.model.pca.n_components

        rec.initialise_output_grid(gt_grid=sc.gt_tsdf)
        oracle_voxlets = rec.fill_in_output_grid_oma(
            render_type=[],oracle='pca', add_ground_plane=True, feature_collapse_type='pca', use_binary=parameters.use_binary)

        tic= time()
        sc.gt_tsdf.render_view(gt_renderpath % (sequence['name'], 'gt'))
        print "One render: ", time() - tic

        sc.im_tsdf.render_view(gt_renderpath % (sequence['name'], 'im'))

        oracle_voxlets.render_view(gen_renderpath % (sequence['name'
            ], num_components))