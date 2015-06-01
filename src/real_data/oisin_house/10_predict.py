'''
make predictions for all of bigbird dataset
using my algorhtm
saves each prediction to disk
'''
import numpy as np
import cPickle as pickle
import sys
import os
sys.path.append(os.path.expanduser("~/projects/shape_sharing/src/"))
from time import time
import yaml
import functools

import real_data_paths as paths
from common import voxlets, scene
import system_setup

parameters_path = './testing_params.yaml'
parameters = yaml.load(open(parameters_path))

render_top_view = True

if __name__ == '__main__':

    # loop over each test time in the testing parameters:
    for params in parameters['tests']:

        print "--> DOING TEST: ", params['name']

        print "--> Loading models..."
        models = [pickle.load(open(paths.voxlet_model_path % name))
                  for name in params['models_to_use']]
        print models[0].forest

        def process_sequence(sequence):

            print "-> Loading ", sequence['name']
            sc = scene.Scene(params['mu'], [])
            sc.load_sequence(
                sequence, frame_nos=0, segment_with_gt=True, voxel_normals='gt_tsdf')
            sc.sample_points(params['number_samples'])

            print "-> Creating folder"
            fpath = paths.prediction_folderpath % \
                (parameters['batch_name'], sequence['name'])
            if not os.path.exists(fpath):
                os.makedirs(fpath)

            if 'ground_truth' in params and params['ground_truth']:
                pred_voxlets = sc

            elif 'visible' in params and params['visible']:
                pred_voxlets = sc.im_tsdf

            else:
                print "-> Setting up the reconstruction object"
                rec = voxlets.Reconstructer()
                rec.set_scene(sc)
                rec.initialise_output_grid(gt_grid=sc.gt_tsdf)
                rec.set_probability_model_one(0.5)
                rec.set_model(models)

                for model in rec.model:
                    model.reset_voxlet_counts()
                    model.set_max_depth(params['max_depth'])

                print "-> Doing prediction, type ", params['name']
                # parameters from the yaml file are passed as separate arguments to voxlets
                pred_voxlets = rec.fill_in_output_grid(**params['reconstruction_params'])

                print "-> Saving the sampled_idxs to a file"
                np.savetxt(fpath + 'sampled_idxs.csv', sc.sampled_idxs, delimiter=",")

                if render_top_view:
                    print "-> Rendering top view"
                    gen_renderpath = paths.voxlet_prediction_img_path % \
                        (parameters['batch_name'], sequence['name'], '%s')
                    rec.plot_voxlet_top_view(savepath=gen_renderpath % 'top_view')

            prediction_savepath = fpath + params['name'] + '.pkl'
            print "-> Saving the prediction to ", prediction_savepath

            with open(prediction_savepath, 'w') as f:
                pickle.dump(pred_voxlets, f, protocol=pickle.HIGHEST_PROTOCOL)

        print "--> Doing test type ", params['name']
        tic = time()

        if system_setup.multicore:
            # need to import this *after* the pool helper has been defined
            import multiprocessing
            pool = multiprocessing.Pool(system_setup.testing_cores)
            pool.map_async(process_sequence, paths.test_data).get(9999999)
            pool.close()
            pool.join()
        else:
            map(process_sequence, paths.test_data)

        print "This test took %f s" % (time() - tic)
