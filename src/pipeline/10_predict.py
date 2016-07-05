'''
make predictions for all of bigbird dataset
using my algorhtm
saves each prediction to disk
'''
import numpy as np
import cPickle as pickle
import sys
import os
import shutil
from time import time
import yaml
import functools
from copy import deepcopy
sys.path.append('..')
from common import voxlets, scene
import system_setup

if len(sys.argv) > 1:
    parameters_path = sys.argv[1]
else:
    parameters_path = './testing_params.yaml'
parameters = yaml.load(open(parameters_path))

if parameters['testing_data'] == 'oisin_house':
    import real_data_paths as paths
elif parameters['training_data'] == 'nyu_cad_silberman':
    import nyu_cad_paths_silberman as paths
else:
    raise Exception('Unknown training data')

print len(paths.test_data)
render_top_view = True


if __name__ == '__main__':

    # loop over each test time in the testing parameters:
    for params in parameters['tests']:

        print "--> DOING TEST: ", params['name']

        print "--> Loading models..."
        vox_model_path = paths.voxlet_model_path
        print [vox_model_path % name for name in params['models_to_use']]
        models = [pickle.load(open(vox_model_path % name))
                  for name in params['models_to_use']]

        def process_sequence(sequence):
            print "-> Creating folder"
            fpath = paths.prediction_folderpath % \
                (parameters['batch_name'], sequence['name'])
            if not os.path.exists(fpath):
                os.makedirs(fpath)

            prediction_savepath = fpath + params['name'] + '.pkl'
            if os.path.exists(prediction_savepath) and len(sys.argv) > 2:
                print "Skipping"
                return

            print "-> Loading ", sequence['name']
            sc = scene.Scene(params['mu'], [])
            sc.load_sequence(
                sequence, frame_nos=0, segment_with_gt=False,
                segment=False, original_nyu=parameters['original_nyu'])
            sc.sample_points(params['number_samples'],
                nyu='nyu_cad' in parameters['testing_data'])
            sc.im._clear_cache()

            if 'ground_truth' in params and params['ground_truth']:
                # import pdb; pdb.set_trace()
                sc2 = deepcopy(sc)
                sc2.gt_labels = []
                sc2.im_visible = []
                sc2.gt_tsdf_separate = []
                sc2.gt_labels_separate = []
                pred_voxlets = sc2

            elif 'visible' in params and params['visible']:
                pred_voxlets = sc.im_tsdf

            else:
                tic = time()
                print "-> Setting up the reconstruction object"
                rec = voxlets.Reconstructer()
                rec.set_scene(sc)
                rec.initialise_output_grid(gt_grid=sc.gt_tsdf,
                    keep_explicit_count=params['reconstruction_params']['weight_predictions'])
                rec.set_model_probabilities(params['model_probabilities'])
                rec.set_model(models)
                rec.mu = params['mu']

                for model in rec.model:
                    model.reset_voxlet_counts()
                    model.set_max_depth(params['max_depth'])
                print "-> Doing prediction, type ", params['name']
                # parameters from the yaml file are passed as separate arguments to voxlets
                pred_voxlets = rec.fill_in_output_grid(**params['reconstruction_params'])
                print "TOOK %f seconds" % (time() - tic)

                print "-> Saving the sampled_idxs to a file"
                np.savetxt(fpath + 'sampled_idxs.csv', sc.sampled_idxs, delimiter=",")

                if render_top_view:
                    print "-> Rendering top view"
                    gen_renderpath = paths.voxlet_prediction_img_path % \
                        (parameters['batch_name'], sequence['name'], '%s')
                    rec.plot_voxlet_top_view(savepath=gen_renderpath % 'top_view')

                # save the keeping existing prediction
                if hasattr(rec, 'keeping_existing'):
                    prediction_savepath = fpath + params['name'] + '.pkl'
                    with open(prediction_savepath.replace('.pkl', '_keeping_existing.pkl'), 'w') as f:
                        pickle.dump(rec.keeping_existing, f, -1)

            print "-> Saving the prediction to ", prediction_savepath
            print "-> Copying the ground truth "
            shutil.copy(sequence['folder'] + sequence['scene'] + '/ground_truth_tsdf.mat',
                        fpath + 'ground_truth.mat')

            with open(prediction_savepath, 'w') as f:
                pickle.dump(pred_voxlets, f, protocol=pickle.HIGHEST_PROTOCOL)

            print "-> Saving the voxlet counts"
            rec.save_voxlet_counts(fpath + 'voxlet_counts.csv')

        print "--> Doing test type ", params['name']

        tic = time()
        # print "WARNING - smallsequence" * 10
        # paths.test_data = paths.test_data

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
