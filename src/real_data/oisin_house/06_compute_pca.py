'''
Loads in the combined sboxes and clusters them to form a dictionary
smaller
'''
import numpy as np
import cPickle as pickle
import sys
import os
from time import time
import yaml
import functools
from sklearn.decomposition import RandomizedPCA

sys.path.append(os.path.expanduser('~/projects/shape_sharing/src/'))
from common import scene, rendering

import system_setup

parameters = yaml.load(open('./training_params.yaml'))

if parameters['training_data'] == 'oisin_house':
    import real_data_paths as paths
elif parameters['training_data'] == 'synthetic':
    import synthetic_paths as paths
else:
    raise Exception('Unknown training data')

# Only using a subset of training sequences
# Ensuring to randomly sample from them, in case there is some sort of inherent
# ordering in the training data
if parameters['pca']['max_sequences'] < len(paths.all_train_data):
    np.random.seed(10)
    train_data_to_use = np.random.choice(
        paths.all_train_data, parameters['pca']['max_sequences'], replace=False)
else:
    train_data_to_use = paths.all_train_data


def flatten_sbox(sbox):
    return sbox.V.flatten()


def fit_and_save_pca(np_array, savepath):

    if parameters['pca']['subsample_length'] < np_array.shape[0]:
        idxs = np.random.choice(
            np_array.shape[0], parameters['pca']['subsample_length'], replace=False)
        np_array = np_array[idxs]

    # fit the pca model
    pca = RandomizedPCA(n_components=parameters['pca']['number_dims'])
    pca.fit(np_array)

    with open(savepath, 'wb') as f:
        pickle.dump(pca, f, pickle.HIGHEST_PROTOCOL)


def process_sequence(sequence, voxlet_params):

    if not os.path.exists(sequence['folder'] + sequence['scene'] + '/ground_truth_tsdf.pkl'):
        print "Failed"
        return

    print "--> Processing " + sequence['scene']
    sc = scene.Scene(parameters['mu'], voxlet_params)
    sc.load_sequence(sequence, frame_nos=0, segment_with_gt=True, voxel_normals='gt_tsdf')

    # sampling points and extracting voxlets at these locations
    idxs = sc.sample_points(parameters['pca']['number_points_from_each_image'],
                      additional_mask=sc.gt_im_label != 0)
    gt_shoeboxes = [sc.extract_single_voxlet(
        idx, extract_from='gt_tsdf', post_transform=flatten_sbox) for idx in idxs]

    return np.array(gt_shoeboxes)



def extract_all_voxlets(voxlet_params_in):

    # this allows for passing multiple arguments to the mapper
    func = functools.partial(process_sequence, voxlet_params=voxlet_params_in)

    # need to import these *after* the pool helper has been defined
    if system_setup.multicore:
        import multiprocessing
        pool = multiprocessing.Pool(system_setup.cores)
        voxlet_list = pool.map(func, train_data_to_use)
        pool.close()
        pool.join()
    else:
        map(func, train_data_to_use)

    print "length is ", len(voxlet_list)
    np_voxlets = np.vstack(voxlet_list).astype(np.float16)

    print "-> Shoeboxes are shape " + str(np_voxlets.shape)
    return np_voxlets


def render_some_voxlets(np_voxlets, np_masks, voxlet_params, folderpath):
    # render views of some of the extracted voxlets to a folder for inspection

    num_to_render = 20
    idxs = np.random.choice(np_voxlets.shape[0], num_to_render)

    if not os.path.exists(folderpath):
        os.makedirs(folderpath)

    for count, idx in enumerate(idxs):
        # doing 3d render
        arrs = (np_voxlets[idx].reshape(voxlet_params['shape']),
                np_masks[idx].reshape(voxlet_params['shape']))

        height = voxlet_params['name'].partition("_")[0]
        savepath = folderpath + '/%03d_voxlet.png' % count
        rendering.render_single_voxlet(
            arrs[0], savepath, height=height, speed='quick')

        # saving the slice
        savepath = folderpath + '/%03d_slice.png' % count
        rendering.plot_slices(arrs, savepath, titles=['TSDF', 'Mask'])


if __name__ == '__main__':

    # Repeat for each type of voxlet in the parameters
    for voxlet_name, voxlet_params in parameters['voxlet_sizes'].iteritems():

        tic = time()

        pca_savefolder = paths.voxlets_dictionary_path % voxlet_name

        if not os.path.exists(pca_savefolder):
            os.makedirs(pca_savefolder)

        print "-> Extracting the voxlets, type %s" % voxlet_name
        np_voxlets = extract_all_voxlets(voxlet_params)

        print "-> Extracting masks"
        np_masks = np.isnan(np_voxlets).astype(np.float16)
        np_voxlets[np_masks == 1] = parameters['mu']

        print "-> Rendering"
        folderpath = paths.voxlets_dictionary_path + '/renders/' + voxlet_name + '/'
        render_some_voxlets(np_voxlets, np_masks, voxlet_params, folderpath)

        print "-> Doing the PCA"
        fit_and_save_pca(np_voxlets, pca_savefolder + 'voxlets_pca.pkl')
        fit_and_save_pca(np_masks, pca_savefolder + 'masks_pca.pkl')

        del np_voxlets, np_masks

        print "In total took %f s" % (time() - tic)