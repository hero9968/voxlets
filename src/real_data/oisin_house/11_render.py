
import numpy as np
import cPickle as pickle
import sys
import os
import yaml
import system_setup
import scipy.misc
import matplotlib.pyplot as plt
sys.path.append(os.path.expanduser("~/projects/shape_sharing/src/"))
from common import voxlets, scene

parameters_path = './testing_params_nyu.yaml'
parameters = yaml.load(open(parameters_path))

if parameters['testing_data'] == 'oisin_house':
    import real_data_paths as paths
elif parameters['testing_data'] == 'synthetic':
    import synthetic_paths as paths
elif parameters['testing_data'] == 'nyu_cad':
    import nyu_cad_paths as paths
else:
    raise Exception('Unknown training data')


# options for rendering
render_top_view = False
combining_renders = True

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


def get_best_subplot_dims(N):
    if N == 1:
        return 1, 1
    elif N == 2:
        return 1, 2
    elif N == 3:
        return 1, 3
    else:
        return np.floor(np.sqrt(N)), np.floor(np.sqrt(N))

def combine_renders(combines, gen_renderpath, savename):

    fig = plt.figure(figsize=(25, 10), dpi=1000)
    plt.subplots(su, sv)
    plt.subplots_adjust(left=0, bottom=0, right=0.98, top=0.98, wspace=0.02, hspace=0.02)

    su, sv = get_best_subplot_dims(len(combines))

    for count, (name, dic) in enumerate(combines.iteritems()):

        plt.subplot(su, sv, count + 1)
        plt.imshow(scipy.misc.imread(gen_renderpath % name))
        plt.axis('off')
        plt.title(dic['name'], fontsize=10)

        " Add to the roc plot, which is in the final subplot"
        if name != 'input':
            "Add the AUC"
            plt.text(0, 50, "AUC=%0.3f, prec=%0.3f, rec=%0.3f" % \
                (dic['results']['auc'],
                 dic['results']['precision'],
                 dic['results']['recall']),
                fontsize=6, color='white')

        else:
            plt.hold(1)
            # plt.plot(rec.sampled_idxs[:, 1], rec.sampled_idxs[:, 0], 'r.', ms=2)

    plt.savefig(savename, dpi=400)
    plt.close()


def slice_render(prediction, gt, savepath):
    # Should save a slice to disk...
    height = 20

    plt.subplot(121)
    plt.imshow(prediction.V[:, :, height], cmap=plt.get_cmap('bwr'))
    plt.clim(-0.025, 0.025)

    plt.subplot(122)
    plt.imshow(gt.V[:, :, height], cmap=plt.get_cmap('bwr'))
    plt.clim(-0.025, 0.025)

    plt.savefig(savepath)
    plt.close()


def process_sequence(sequence):


    print "-> Loading ground truth", sequence['name']
    fpath = paths.prediction_folderpath % (parameters['batch_name'], sequence['name'])
    gt_scene = pickle.load(open(fpath + 'ground_truth.pkl'))

    # Path where any renders will be saved to
    gen_renderpath = paths.voxlet_prediction_img_path % \
        (parameters['batch_name'], sequence['name'], '%s')

    print "-> Saving the input rgb, depth and mask"
    write_input_images(gt_scene, gen_renderpath)

    print "-> Main renders"
    for test_params in parameters['tests']:

            # if test_params['name'] == 'ground_truth':
            #     continue

            print "Loading ", test_params['name']

            prediction_savepath = fpath + test_params['name'] + '.pkl'

            if not os.path.exists(prediction_savepath):
                continue

            prediction = pickle.load(open(prediction_savepath))

            if test_params['name'] == 'ground_truth':
                prediction = prediction.gt_tsdf

            if test_params['name']=='visible':
                ground_height = None
            else:
                ground_height = 0.03

            # sometimes multiple predictions are stored in predicton
            print "Rendering ", test_params['name']
            if hasattr(prediction, '__iter__'):
                for key, item in prediction.iteritems():
                    savepath = (gen_renderpath % test_params['name']).replace('.png', str(key) + '.png')
                    item.render_view(savepath, xy_centre=True, ground_height=ground_height, keep_obj=True)
                    print "Saving to ", savepath
            else:
                prediction.render_view(gen_renderpath % test_params['name'],
                    xy_centre=True, ground_height=ground_height, keep_obj=True)
                print "Saving to ", gen_renderpath % test_params['name']

            # maybe I also want to render other types of prediction
            if parameters['render_without_excess_removed']:

                print "Rendering without excess removed"
                prediction_savepath = fpath + test_params['name'] + '_average.pkl'

                if not os.path.exists(prediction_savepath):
                    continue

                prediction = pickle.load(open(prediction_savepath))

                print "Saving to ", gen_renderpath % test_params['name']
                prediction.render_view(gen_renderpath % test_params['name'],
                    xy_centre=True, ground_height=ground_height, keep_obj=True)

                # savepath = (gen_renderpath % test_params['name']).replace('.png', '_slice.png')
                # slice_render(prediction, gt_scene.gt_tsdf, savepath)
                #



    # if combining_renders:
    #     print "-> Combining renders"
    #     su, sv = 3, 3

    #     all_savename = gen_renderpath.replace('png', 'pdf') % ('all_' + sequence['name'])
    #     combine_renders(combines, su, sv, gen_renderpath, all_savename)


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




# print "Done loading model..."

# cobweb = True
# just_render_gt = False  # this overwrites the existing
# render_gt = False
# combining_renders = False
# render_predictions = True
# render_top_view = False
# do_greedy_oracles = False
# # save_prediction_grids = True
# save_simple = True
# save_scores_to_yaml = True
# distance_experiments = False


# copy_to_dropbox = False and paths.host_name == 'biryani'
# dropbox_path = paths.new_dropbox_dir()
# print dropbox_path


# test_type = 'different_data_split_dw_empty'



    # print "Warning - just doing this one"
    # this_one = []
    # for t in paths.test_data:
    #     if t['name'] == 'saved_00196_[92]':
    #         this_one.append(t)
    # temp = [s for s in paths.RenderedData.test_sequence() if s['name'] == 'd2p8ae7t0xi81q3y_SEQ']
    # print temp
