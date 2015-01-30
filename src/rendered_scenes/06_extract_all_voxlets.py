'''
Extracts all the shoeboxes from all the training images
'''
import numpy as np
import cPickle as pickle
import sys
import os
from time import time
import scipy.io

sys.path.append('/Users/Michael/projects/shape_sharing/src/')
from common import paths
from common import voxel_data
from common import images
from common import parameters
from common import features

pca_savepath = paths.RenderedData.voxlets_dictionary_path + 'pca.pkl'
with open(pca_savepath, 'rb') as f:
    pca = pickle.load(f)

if not os.path.exists(paths.RenderedData.voxlets_data_path):
    os.makedirs(paths.RenderedData.voxlets_data_path)


def pool_helper(index, im, vgrid):

    world_xyz = im.get_world_xyz()
    world_norms = im.get_world_normals()

    # convert to linear idx
    point_idx = index[0] * im.mask.shape[1] + index[1]

    shoebox = voxel_data.ShoeBox(parameters.Voxlet.shape, np.float32)
    shoebox.V *= 0
    shoebox.V += parameters.RenderedVoxelGrid.mu  # set the outside area to mu
    shoebox.set_p_from_grid_origin(parameters.Voxlet.centre)  # m
    shoebox.set_voxel_size(parameters.Voxlet.size)  # m
    shoebox.initialise_from_point_and_normal(
        world_xyz[point_idx], world_norms[point_idx], np.array([0, 0, 1]))

    # convert the indices to world xyz space
    shoebox.fill_from_grid(vgrid)

    # convert to pca representation
    pca_representation = pca.transform(shoebox.V.flatten())
    # pca_kmeans_idx = km_pca.predict(pca_representation.flatten())
    # kmeans_idx = km_standard.predict(shoebox.V.flatten())

    all_representations = dict(pca_representation=pca_representation)
    #                           pca_kmeans_idx=pca_kmeans_idx,
    #                           kmeans_idx=kmeans_idx)

    # print a dot each time the function is run, without a new line
    sys.stdout.write('.')
    sys.stdout.flush()
    return all_representations

# need to import these *after* the pool helper has been defined
if parameters.multicore:
    import multiprocessing
    import functools
    pool = multiprocessing.Pool(parameters.cores)

for count, sequence in enumerate(paths.RenderedData.train_sequence()):

    print "Processing " + sequence['name']

    # load in the ground truth grid for this scene, and converting nans
    vox_dir = paths.RenderedData.ground_truth_voxels(sequence['scene'])
    gt_vox = voxel_data.load_voxels(vox_dir)
    gt_vox.V[np.isnan(gt_vox.V)] = -parameters.RenderedVoxelGrid.mu
    gt_vox.set_origin(gt_vox.origin)

    # loading this frame
    frame_data = paths.RenderedData.load_scene_data(
        sequence['scene'], sequence['frames'][0])
    im = images.RGBDImage.load_from_dict(
        paths.RenderedData.scene_dir(sequence['scene']), frame_data)

    # computing normals...
    norm_engine = features.Normals()
    im.normals = norm_engine.compute_normals(im)

    "Sampling from image"
    idxs = im.random_sample_from_mask(
        parameters.VoxletTraining.number_points_from_each_image)

    "Extracting features"
    ce = features.CobwebEngine(
        t=parameters.VoxletTraining.cobweb_t, fixed_patch_size=False)
    ce.set_image(im)
    np_features = np.array(ce.extract_patches(idxs))

    "Now try to make this nice and like parrallel or something...?"
    t1 = time()
    if parameters.multicore:
        shoeboxes = pool.map(
            functools.partial(pool_helper, im=im, vgrid=gt_vox), idxs)
    else:
        shoeboxes = [pool_helper(idx, im=im, vgrid=gt_vox) for idx in idxs]
    print "Took %f s" % (time() - t1)

    np_sboxes = np.array(shoeboxes)

    print "Shoeboxes are shape " + str(np_sboxes.shape)
    print "Features are shape " + str(np_features.shape)

    #   D = dict(shoeboxes=np_sboxes, features=np_features)
    savepath = paths.RenderedData.voxlets_data_path + \
        sequence['name'] + '.mat'
    print savepath
    #    scipy.io.savemat(savepath, D, do_compression=True)

    # if count > 8 and parameters.small_sample:
    #     print "Ending now"
    #     break

    print "np+features shape is", np_features.shape

    # convert the shoeboxes to individual components
    np_pca_representation = np.array(
        [sbox['pca_representation'] for sbox in shoeboxes])
    np_pca_representation = np_pca_representation.reshape(
        (-1, np_pca_representation.shape[2]))
    #    np_kmeans_idx = np.array(
    #        [sbox['kmeans_idx'] for sbox in shoeboxes]).flatten()
    #    np_pca_kmeans_idx = np.array(
    #        [sbox['pca_kmeans_idx'] for sbox in shoeboxes]).flatten()

    print "PCA is shape " + str(np_pca_representation.shape)
#    print "kmeans is shape " + str(np_kmeans_idx.shape)
#    print "pca kmeans is shape " + str(np_pca_kmeans_idx.shape)

    print "Features are shape " + str(np_features.shape)
    D = dict(pca_representation=np_pca_representation,
             features=np_features)
#             pca_kmeans_idx=np_pca_kmeans_idx,
#             kmeans_idx=np_kmeans_idx,

    scipy.io.savemat(savepath, D, do_compression=True)

    if count > parameters.max_sequences:
        print "Ending now"
        break
