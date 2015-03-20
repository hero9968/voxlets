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

from common import paths
from common import parameters
from common import voxlets
from common import scene


# loading model
with open(paths.RenderedData.voxlet_model_oma_path, 'rb') as f:
    model = pickle.load(f)

# for count, tree in enumerate(model.forest.trees):
#     print count, len(tree.leaf_nodes())
#     lengths = np.array([len(leaf.exs_at_node) for leaf in tree.leaf_nodes()])
#     print np.sum(lengths<10), np.sum(np.logical_and(lengths>10, lengths<50)), np.sum(lengths>50)


test_types = ['oma_implicit']

# print "Checking results folders exist, creating if not"
# for test_type in test_types + ['partial_tsdf', 'visible_voxels']:
#     print test_type
#     folder_save_path = \
#         paths.RenderedData.voxlet_prediction_path % (test_type, '_')
#     folder_save_path = os.path.dirname(folder_save_path)
#     print folder_save_path
#     if not os.path.exists(folder_save_path):
#         os.makedirs(folder_save_path)


print "MAIN LOOP"
# Note: if parallelising, should either do here (at top level) or at the
# bottom level, where the predictions are accumulated (although this might be)
# better off being GPU...)
def process_sequence(sequence):

    sc = scene.Scene()
    sc.load_sequence(sequence, frame_nos=0, segment_with_gt=True, save_grids=False, load_implicit=True)
    sc.santity_render(save_folder='/tmp/')

    test_type = 'oma_implicit'

    print "-> Reconstructing with oma forest"
    rec = voxlets.Reconstructer(
        reconstruction_type='kmeans_on_pca', combine_type='modal_vote')
    rec.set_model(model)
    rec.set_scene(sc)
    rec.sample_points(parameters.VoxletPrediction.number_samples)
    rec.initialise_output_grid(gt_grid=sc.gt_tsdf)
    accum = rec.fill_in_output_grid_oma( render_type=[], #['matplotlib'],
        render_savepath='/tmp/renders/')
    prediction = accum
#    prediction = accum.compute_average(
 #       nan_value=parameters.RenderedVoxelGrid.mu)
    prediction_keeping_exisiting = rec.keeping_existing

    # Hack to put in a floor
    prediction_keeping_exisiting.V[:, :, :4] = -1
    prediction.V[:, :, :4] = -1


    print "-> Creating folder"
    fpath = paths.RenderedData.voxlet_prediction_folderpath % \
        (test_type, sequence['name'])
    if not os.path.exists(fpath):
        os.makedirs(fpath)

    print "-> Saving"
    # savepath = paths.RenderedData.voxlet_prediction_path % \
    #     (test_type, sequence['name'])
    # prediction.save(savepath)
    # savepath = paths.RenderedData.voxlet_prediction_path % \
    #     (test_type, sequence['name'], 'keep_existing')
    # prediction_keeping_exisiting.save(savepath)

    print "-> Rendering"
    gen_renderpath = paths.RenderedData.voxlet_prediction_img_path % \
        (test_type, sequence['name'], '%s')

    
    prediction_keeping_exisiting.render_view(gen_renderpath % 'keep_existing')
    prediction.render_view(gen_renderpath % 'not_keeping_existing')
    sc.implicit_tsdf.render_view(gen_renderpath % 'implicit')
    sc.im_tsdf.render_view(gen_renderpath % 'visible')
    sc.gt_tsdf.render_view(gen_renderpath % 'gt')
    scipy.misc.imsave(gen_renderpath % 'input', sc.im.rgb)

    combines = (
        ('Ground truth', 'gt'),
        ('Input image', 'input'),
        ('Visible surfaces', 'visible'),
        ('Implicit prediction', 'implicit'),
        ('Voxlets', 'not_keeping_existing'),
        ('Voxlets with visible surfaces', 'keep_existing'))

    su, sv = 2, 3

    fig = plt.figure(figsize=(20, 10))
    plt.subplots(su, sv)
    plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=1e-3, hspace=1e-3)

    for count, c in enumerate(combines):
        plt.subplot(su, sv, count + 1)
        plt.imshow(scipy.misc.imread(gen_renderpath % c[1]))
        plt.axis('off')
        plt.title(c[0])

    plt.savefig(gen_renderpath.replace('png', 'pdf') % 'all')

    print "-> Done test type " + test_type

    print "Done sequence %s" % sequence['name']


# need to import these *after* the pool helper has been defined
if False:  # parameters.multicore:
    import multiprocessing
    pool = multiprocessing.Pool(parameters.cores)
    mapper = pool.map
else:
    mapper = map


if __name__ == '__main__':

    tic = time()
    mapper(process_sequence, paths.RenderedData.test_sequence())
    print "In total took %f s" % (time() - tic)


