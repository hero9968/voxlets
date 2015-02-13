'''
Similar to 08_predict, but uses an oracle to choose the best [training
voxlet/kmeans cluster centre] to fit to the scene, instead of using the
forest. This should show us the 'best case' scenario, assuming a perfect
forest lookup...
'''
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cPickle as pickle
import sys
import os
sys.path.append(os.path.expanduser("~/projects/shape_sharing/src/"))

from common import paths
from common import parameters
from common import voxel_data
from common import images
from common import features
from common import voxlets
from common import carving
from common import mesh


test_types = ['tall_gt_oracle']

print "Checking results folders exist, creating if not"
for test_type in test_types + ['partial_tsdf', 'visible_voxels']:
    print test_type
    folder_save_path = \
        paths.RenderedData.voxlet_prediction_path % (test_type, '_')
    folder_save_path = os.path.dirname(folder_save_path)
    print folder_save_path
    if not os.path.exists(folder_save_path):
        os.makedirs(folder_save_path)


def extract_shoebox(index, im, vgrid):

    world_xyz = im.get_world_xyz()
    world_norms = im.get_world_normals()

    # convert to linear idx
    point_idx = index[0] * im.mask.shape[1] + index[1]

    shoebox = voxel_data.ShoeBox(parameters.Voxlet.shape, np.float32)
    shoebox.V *= 0
    shoebox.V -= parameters.RenderedVoxelGrid.mu  # set the outside area to -mu
    shoebox.set_p_from_grid_origin(parameters.Voxlet.centre)  # m
    shoebox.set_voxel_size(parameters.Voxlet.size)  # m

    start_x = world_xyz[point_idx, 0]
    start_y = world_xyz[point_idx, 1]
    start_z = parameters.Voxlet.centre[2]
    shoebox.initialise_from_point_and_normal(
        np.array([start_x, start_y, start_z]),
        world_norms[point_idx],
        np.array([0, 0, 1]))

    # convert the indices to world xyz space
    shoebox.fill_from_grid(vgrid)

    sys.stdout.write('.')
    sys.stdout.flush()

    return shoebox.V.flatten()


def reconstruct_grid(idxs, im, blank_vox, sboxes):
    '''
    reconstructs the grid, given a blank grid and an image and a thing to put
    at each location in the image
    '''
    assert len(idxs) == len(sboxes)

    world_xyz = im.get_world_xyz()
    world_norms = im.get_world_normals()

    # TODO - should really use this accumulator...
    # self.accum = voxel_data.UprightAccumulator(gt_grid.V.shape)
    # self.accum.set_origin(gt_grid.origin)
    # self.accum.set_voxel_size(gt_grid.vox_size)

    for idx, sbox in zip(idxs, sboxes):

        # convert to linear idx
        point_idx = idx[0] * im.mask.shape[1] + idx[1]

        shoebox = voxel_data.ShoeBox(parameters.Voxlet.shape, np.float32)
        shoebox.set_p_from_grid_origin(parameters.Voxlet.centre)  # m
        shoebox.set_voxel_size(parameters.Voxlet.size)  # m

        start_x = world_xyz[point_idx, 0]
        start_y = world_xyz[point_idx, 1]
        start_z = parameters.Voxlet.centre[2]
        shoebox.initialise_from_point_and_normal(
            np.array([start_x, start_y, start_z]),
            world_norms[point_idx],
            np.array([0, 0, 1]))

        shoebox.V = sbox.reshape(shoebox.V.shape)

        blank_vox.add_voxlet(shoebox)

        sys.stdout.write('.')
        sys.stdout.flush()

    return blank_vox.compute_average(nan_value = parameters.RenderedVoxelGrid.mu)


print "MAIN LOOP"
# Note: if parallelising, should either do here (at top level) or at the
# bottom level, where the predictions are accumulated (although this might be)
# better off being GPU...)
for count, sequence in enumerate(paths.RenderedData.test_sequence()):

    # load in the ground truth grid for this scene, and converting nans
    vox_location = paths.RenderedData.ground_truth_voxels(sequence['scene'])
    gt_vox = voxel_data.load_voxels(vox_location)
    gt_vox.V[np.isnan(gt_vox.V)] = -parameters.RenderedVoxelGrid.mu
    gt_vox.set_origin(gt_vox.origin)

    # loading in the image
    frame_data = paths.RenderedData.load_scene_data(
        sequence['scene'], sequence['frames'][0])
    im = images.RGBDImage.load_from_dict(
        paths.RenderedData.scene_dir(sequence['scene']),
        frame_data)

    # computing normals...
    norm_engine = features.Normals()
    im.normals = norm_engine.compute_normals(im)

    # while I'm here - might as well save the image as a voxel grid
    video = images.RGBDVideo.init_from_images([im])
    carver = carving.Fusion()
    carver.set_video(video)
    carver.set_voxel_grid(gt_vox.blank_copy())
    partial_tsdf, visible = carver.fuse(parameters.RenderedVoxelGrid.mu)

    # # save this as a voxel grid...
    render_savepath = paths.RenderedData.voxlet_prediction_img_path % \
            ('partial_tsdf', sequence['name'])
    partial_tsdf.render_view(render_savepath)

    print "Extracting the ground truth voxlets"
    idxs = im.random_sample_from_mask(
        parameters.VoxletTraining.number_points_from_each_image)

    sboxes = [extract_shoebox(idx, im, gt_vox) for idx in idxs]


    ##############################################################
    test_type = 'kmeans_oracle'
    print "Performing test type...", test_type
    ##############################################################

    kmeans_savepath = paths.RenderedData.voxlets_dictionary_path + 'kmean.pkl'
    with open(kmeans_savepath, 'rb') as f:
        km = pickle.load(f)

    kmeans_center_idxs = [km.predict(sbox) for sbox in sboxes]
    kmeans_centers = [
        km.cluster_centers_[c_idx] for c_idx in kmeans_center_idxs]
    print "Centers shape is ", kmeans_centers[0].shape

    print "Reconstructing"
    blank_vox = voxel_data.UprightAccumulator(gt_vox.V.shape)
    blank_vox.origin = gt_vox.origin
    blank_vox.R = gt_vox.R
    blank_vox.vox_size = gt_vox.vox_size
    km_predict = reconstruct_grid(idxs, im, blank_vox, kmeans_centers)

    print "Saving"
    savepath = paths.RenderedData.voxlet_prediction_path % \
        (test_type, sequence['name'])
    km_predict.save(savepath)

    print "Rendering"
    render_savepath = paths.RenderedData.voxlet_prediction_img_path % \
        (test_type, sequence['name'])
    km_predict.render_view(render_savepath)

    print "Done test type " + test_type

    ##############################################################
    test_type = 'gt_oracle'
    print "Performing test type...", test_type
    ##############################################################

    print "Now reconstructing with the ground truth shoeboxes"
    blank_vox = voxel_data.UprightAccumulator(gt_vox.V.shape)
    #blank_vox.V = np.zeros(gt_vox.V.shape) + parameters.RenderedVoxelGrid.mu
    blank_vox.origin = gt_vox.origin
    blank_vox.R = gt_vox.R
    blank_vox.vox_size = gt_vox.vox_size
    gt_predict = reconstruct_grid(idxs, im, blank_vox, sboxes)

    print "Saving"
    savepath = paths.RenderedData.voxlet_prediction_path % \
        (test_type, sequence['name'])
    gt_predict.save(savepath)

    print "Rendering"
    render_savepath = paths.RenderedData.voxlet_prediction_img_path % \
        (test_type, sequence['name'])
    gt_predict.render_view(render_savepath)

    print "Done test type " + test_type

    ##############################################################

    print "Done sequence %s" % sequence['name']
    if count >= 3:
        print "BREAKING EARLY"
        break

    # "Saving result to disk"
    # savepath =

    # savepath = paths.voxlet_prediction_path % \
    # (test_type, modelname, test_view)
    # D = dict(prediction=prediction.V, gt=gt)
    # scipy.io.savemat(savepath, D, do_compression=True)

    # "Now also save to a pickle file so have the original data..."
    # savepathpickle = paths.voxlet_prediction_path_pkl % \
    #     (test_type, modelname, test_view)
    # pickle.dump(prediction, open(savepathpickle, 'wb'))

    # "Computing the auc score"
    # gt_occ = ((gt + 0.03) / 0.06).astype(int)
    # prediction_occ = (prediction.V + 0.03) / 0.06
    # auc = sklearn.metrics.roc_auc_score(
    #     gt_occ.flatten(), prediction_occ.flatten())

    # "Filling the figure"
    # imagesavepath = paths.voxlet_prediction_image_path % \
    #     (test_type, modelname, test_view)
    # save_plot_slice(prediction.V, gt, imagesavepath, imtitle=str(auc))

    # # need to do this here after the pool helper has been defined...
    # import multiprocessing
    # import functools

    # if multiproc:
    #     pool = multiprocessing.Pool(parameters.cores)
