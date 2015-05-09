
import numpy as np
import cPickle as pickle
import sys
import os
from time import time
import scipy.io
import logging
logging.basicConfig(level=logging.DEBUG)

sys.path.append(os.path.expanduser('~/projects/shape_sharing/src/'))
from common import scene, voxlets

import real_data_paths as paths
import real_params as parameters

# features_iso_savepath = paths.RenderedData.voxlets_dictionary_path + 'features_iso.pkl'
# with open(features_iso_savepath, 'rb') as f:
#     features_iso = pickle.load(f)

with open(paths.RenderedData.voxlet_model_oma_path) as f:
    model_without_implicit = pickle.load(f)

sequence = paths.train_data[0]

sc = scene.Scene(parameters.mu, parameters.Voxlet)
sc.load_sequence(sequence, frame_nos=0, segment_with_gt=True,
    save_grids=False, load_implicit=False)
# sc.santity_render(save_folder='/tmp/')

# just using reconstructor for sampling the points...
rec = voxlets.Reconstructer(
    reconstruction_type='kmeans_on_pca', combine_type='modal_vote')
rec.set_scene(sc)
rec.sample_points(parameters.VoxletTraining.number_points_from_each_image,
                  parameters.VoxletPrediction.sampling_grid_size,
                  additional_mask=sc.gt_im_label != 0)

idxs = rec.sampled_idxs


rec.initialise_output_grid(gt_grid=sc.gt_tsdf)
rec.set_model(model_without_implicit)

print "-> Rendering top view"
rec.plot_voxlet_top_view(savepath='/tmp/top_view')
