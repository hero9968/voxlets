
import numpy as np
import cPickle as pickle
import sys
import os
import yaml
import system_setup
import scipy.misc
import matplotlib.pyplot as plt
sys.path.append('../..')
from common import voxlets, scene

if len(sys.argv) > 1:
    parameters_path = sys.argv[1]
else:
    parameters_path = './testing_params.yaml'
parameters = yaml.load(open(parameters_path))

if parameters['testing_data'] == 'oisin_house':
    import real_data_paths as paths
else:
    raise Exception('Unknown training data')


# options for rendering
render_keeping_existing = False
do_write_input_images = False
render_the_normal_thing = True


def write_input_images(sc, gen_renderpath):
    scipy.misc.imsave(gen_renderpath % 'input', sc.im.rgb)

    plt.imshow(sc.im.depth)
    plt.axis('off')
    plt.savefig(gen_renderpath % 'input_depth')
    plt.close()

    plt.imshow(sc.im.mask)
    plt.axis('off')
    # plt.plot(rec.sampled_idxs[:, 1], rec.sampled_idxs[:, 0], 'ro')
    plt.savefig(gen_renderpath % 'input_mask')
    plt.close()


def process_sequence(sequence):

    print "-> Loading ground truth", sequence['name']

    fpath = paths.prediction_folderpath % (parameters['batch_name'], sequence['name'])
    if not os.path.exists(fpath + 'ground_truth.pkl'):
        print "Cannot find", fpath
        return
    # gt_scene = pickle.load(open(fpath + 'ground_truth.pkl'))

    # Path where any renders will be saved to
    gen_renderpath = paths.voxlet_prediction_img_path % \
        (parameters['batch_name'], sequence['name'], '%s')

    print "-> Saving the input rgb, depth and mask"
    if do_write_input_images:
        write_input_images(gt_scene, gen_renderpath)

    print "-> Main renders"
    for test_params in parameters['tests']:

            # if test_params['name'] == 'ground_truth':
            #     continue
            if os.path.exists(gen_renderpath % test_params['name']) and \
                len(sys.argv) > 2:
                print "Already done so skipping"
                continue

            print "Loading ", test_params['name']

            prediction_savepath = fpath + test_params['name'] + '.pkl'
            print prediction_savepath
            #
            # if not os.path.exists(prediction_savepath) and test_params['name'] != 'visible':
            #     continue

            if test_params['name']=='visible':
                savepath = gen_renderpath % test_params['name']
                print "Save path is ", savepath
                gt_scene.render_visible(savepath, xy_centre=False, keep_obj=True)
                continue
                # ground_height = None
            else:
                ground_height = 0.0

            if render_the_normal_thing:
                prediction = pickle.load(open(prediction_savepath))

                if test_params['name'] == 'ground_truth':
                    prediction = prediction.gt_tsdf

                # sometimes multiple predictions are stored in predicton
                print "Rendering ", test_params['name']
                if parameters['render_normal']:
                    if hasattr(prediction, '__iter__'):
                        for key, item in prediction.iteritems():
                            savepath = (gen_renderpath % test_params['name']).replace('.png', str(key) + '.png')
                            item.render_view(savepath, xy_centre=True, ground_height=ground_height, keep_obj=True)
                            print "Saving to ", savepath
                    else:
                        prediction.render_view(gen_renderpath % test_params['name'],
                            xy_centre=True, ground_height=0.03,
                            keep_obj=True, actually_render=False,
                            flip=False)
                        print "Saving to ", gen_renderpath % test_params['name']

            # maybe I also want to render other types of prediction
            if render_keeping_existing:

                print "Rendering keeping existing"
                prediction_savepath = fpath + test_params['name'] + '_keeping_existing.pkl'

                if not os.path.exists(prediction_savepath):
                    print "Cannot find!", prediction_savepath
                    continue

                prediction = pickle.load(open(prediction_savepath))

                savepath = (gen_renderpath % test_params['name']) + '_keeping_existing.png'
                print "Saving to ", savepath
                prediction.render_view(
                    savepath,
                    xy_centre=False, ground_height=ground_height, flip=True,
                    keep_obj=True, actually_render=False)

# need to import these *after* the pool helper has been defined
if system_setup.multicore:
    import multiprocessing
    mapper = multiprocessing.Pool(system_setup.testing_cores).map
else:
    mapper = map


if __name__ == '__main__':

    # print "WARNING - SMALL TEST DATA"
    # test_data = yaml.load(open('/media/ssd/data/oisin_house/train_test/test.yaml'))
    test_data = paths.test_data
    results = mapper(process_sequence, test_data)
