'''
make predictions for all of bigbird dataset
using my algorhtm
saves each prediction to disk
'''
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cPickle as pickle
import sys
import os
sys.path.append(os.path.expanduser("~/projects/shape_sharing/src/"))
from time import time
import scipy.misc  # for image saving
import scipy.io
import yaml
import shutil
import collections

import real_data_paths as paths

from common import voxlets
from common import scene

import sklearn.metrics

print "Loading model..."

parameters_path = './testing_params.yaml'
parameters = yaml.load(open(parameters_path))

print "Done loading model..."

cobweb = True
just_render_gt = False  # this overwrites the existing
render_gt = False
combining_renders = False
render_predictions = True
render_top_view = False
do_greedy_oracles = False
# save_prediction_grids = True
save_simple = True
save_scores_to_yaml = True
distance_experiments = False


copy_to_dropbox = False and paths.host_name == 'biryani'
dropbox_path = paths.new_dropbox_dir()
print dropbox_path


test_type = 'different_data_split_dw_empty'


def process_sequence(sequence, params):

    print "-> Loading ", sequence['name']
    sc = scene.Scene(params['mu'], [])
    sc.load_sequence(sequence, frame_nos=0, segment_with_gt=True, voxel_normals='gt_tsdf')
    sc.sample_points(params['number_samples'])

    print "-> Creating folder"
    fpath = paths.voxlet_prediction_folderpath % (test_type, sequence['name'])
    if not os.path.exists(fpath):
        os.makedirs(fpath)
    print "Made folder ", fpath

    print "-> Setting up the reconstruction object"
    rec = voxlets.Reconstructer()
    rec.set_scene(sc)
    rec.initialise_output_grid(gt_grid=sc.gt_tsdf)
    rec.set_probability_model_one(0.5)
    rec.set_model([model_short, model_tall])

    for model in rec.model:
        model.reset_voxlet_counts()
        model.set_max_depth(parameters['max_depth'])

    print "-> Doing prediction, type ", params['type']
    if params['type'] == 'normal':

        pred_voxlets = rec.fill_in_output_grid_oma(
            weight_empty_lower=params['empty_weight'])

        # saving the voxlet counts for each of the models...
        # used for the voxpop analysis
        rec.save_voxlet_counts(fpath)

    elif params['type'] == 'medioid':

        # here I'm doing an experiment to see which distance measure makes sense to use...
        rec.initialise_output_grid(gt_grid=sc.gt_tsdf)
        narrow_band = rec.fill_in_output_grid_oma(
            distance_measure=d_measure,
            weight_empty_lower=params['empty_weight'])

    print "-> Saving the prediction to disk"
    prediction_savepath = paths.prediction_path % (params['name'], sequence['name'])

    here save a yaml version of the test type to the base folder... maybre...


    if do_greedy_oracles:
        print "-> DOING GREEDY ORACLES"

        # true greedy oracle prediction - without the ground truth
        greedy_combines = collections.OrderedDict()
        greedy_combines['gt'] = save_render_assess(
            {'name':'Ground truth', 'grid':sc.gt_tsdf, 'desc':'gt'}, sc)
        greedy_combines['visible'] = save_render_assess(
            {'name':'Visible surfaces', 'grid':sc.im_tsdf, 'desc':'visible'}, sc)

        for d_measure in ['narrow_band', 'largest_of_free_zones', 'just_empty']:

            print "in outer loop" , d_measure
            rec.initialise_output_grid(gt_grid=sc.gt_tsdf)
            true_greedy = rec.fill_in_output_grid_oma(
                render_type=[], add_ground_plane=False, feature_collapse_type='pca',
                oracle='true_greedy', distance_measure=d_measure, cobweb=cobweb)

            for key, result in true_greedy.iteritems():
                print "In innder loop ", key
                dic = {'desc':'greedy_%s_%04d' % (d_measure, key),
                       'name': 'Greedy %s %04d' % (d_measure, key),
                       'grid': result}
                greedy_combines['greedy_%s_%04d' % (d_measure, key)] = \
                    save_render_assess(dic, sc)
            true_greedy = None

        # now combining the greedy renders...
        fname = 'all_greedy_' + rec.sc.sequence['name']
        all_savename = gen_renderpath.replace('png', 'pdf') % fname



# need to import these *after* the pool helper has been defined
if False:
    # parameters.multicore:
    import multiprocessing
    mapper = multiprocessing.Pool(3).map
else:
    mapper = map


if __name__ == '__main__':

    # print "Warning - just doing this one"
    # this_one = []
    # for t in paths.test_data:
    #     if t['name'] == 'saved_00196_[92]':
    #         this_one.append(t)
    # temp = [s for s in paths.RenderedData.test_sequence() if s['name'] == 'd2p8ae7t0xi81q3y_SEQ']
    # print temp

    # loop over each test time in the testing parameters:
    for test_params in parameters.tests:

        tic = time()
        func = functools.partial(process_sequence, params=test_params)
        mapper(process_sequence, paths.test_data)
        print "This test took %f s" % (time() - tic)


