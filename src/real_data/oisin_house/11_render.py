
import numpy as np
import cPickle as pickle
import sys
import os
import yaml
import real_data_paths as paths
import system_setup
import scipy.misc
import matplotlib.pyplot as plt
sys.path.append(os.path.expanduser("~/projects/shape_sharing/src/"))
from common import voxlets, scene

parameters_path = './testing_params.yaml'
parameters = yaml.load(open(parameters_path))


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


def combine_renders(combines, su, sv, gen_renderpath, savename):

    fig = plt.figure(figsize=(25, 10), dpi=1000)
    plt.subplots(su, sv)
    plt.subplots_adjust(left=0, bottom=0, right=0.98, top=0.98, wspace=0.02, hspace=0.02)

    for count, (name, dic) in enumerate(combines.iteritems()):

        if count >= su*sv:
            raise Exception("Error! Final subplot is reserved for the ROC curve")

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

            plt.subplot(su, sv, su*sv)
            plt.plot(dic['results']['fpr'], dic['results']['tpr'], label=dic['name'])
            plt.hold(1)
            plt.legend(prop={'size':6}, loc='lower right')

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

    # if render_top_view:
    #     print "-> Rendering top view"
    #     rec.plot_voxlet_top_view(savepath=gen_renderpath % 'top_view')
    #     print gen_renderpath % 'top_view'

    print "-> Main renders"
    for test_params in parameters['tests']:

            prediction_savepath = fpath + test_params['name'] + '.pkl'
            if os.path.exists(prediction_savepath):

                prediction = pickle.load(open(prediction_savepath))

                savepath = (gen_renderpath % test_params['name']).replace('.png', '_slice.png')
                slice_render(prediction, gt_scene.gt_tsdf, savepath)
                # print "Rendering ", test_params['name']
                # if test_params['name']=='visible':
                #     prediction.render_view(gen_renderpath % test_params['name'],
                #         xy_centre=True)
                # else:
                #     prediction.render_view(gen_renderpath % test_params['name'],
                #         xy_centre=True, ground_height=0.03)

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

    results = mapper(process_sequence, paths.test_data)




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
