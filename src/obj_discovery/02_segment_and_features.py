'''
a script to set up the folders for the training data, also to do things like
the partial kinfu perhaps...
and I think now this should actually do the full thing to extract and save
training data from each sequence
'''
import sys
import os
import scipy.io
import scipy.misc
from time import time
import yaml
import numpy as np
sys.path.append(os.path.expanduser('~/projects/shape_sharing/src/'))
import cPickle as pickle
import system_setup
import paths
from common import scene, features
import matplotlib.pyplot as plt

from skimage.morphology import binary_erosion, binary_dilation, disk

parameters = yaml.load(open('./params.yaml'))

print "Creating output folder"
savefolder = paths.features_dir
if not os.path.exists(savefolder):
    os.makedirs(savefolder)


fe = features.RegionFeatureEngine()
ne = features.Normals()

def process_sequence(sequence):

    save_location = savefolder + sequence['name'] + '.mat'

    print "Processing " + sequence['name']
    sc = scene.Scene(parameters['mu'], None)
    sc.load_sequence(sequence, frame_nos=0, segment_with_gt=True,
        segment=True, save_grids=False, carve=False)

    # want to load in the pre-computed 2d labelling
    matpath = paths.labels_dir + sequence['scene'] + '.mat'
    # print sequence
    # print matpath
    labels = scipy.io.loadmat(matpath)['seg']
    el = disk(3)
    for idx in range(1, labels.max()+1):
        labels[binary_dilation(labels == idx, el)] = idx
    print labels.shape, sc.gt_tsdf.V.shape, sc.temp_2d_labels.shape

    # then stack it up tall
    labels_3d = sc.gt_tsdf.blank_copy()
    labels_3d.V = np.tile(labels[:, :, None], [1, 1, sc.gt_tsdf.V.shape[2]]).astype(float)
    print "3d shape is ", labels_3d.V.shape, sc.gt_tsdf.V.shape

    # then 'and' it with the occupied voxels
    labels_3d.V[sc.gt_tsdf.V < 0] = np.nan

    # finally reproject into the image to get the image labelling
    im_label = sc.im.label_from_grid(labels_3d)
    im_label[im_label==0] = np.nan

    plt.imshow(im_label)
    plt.savefig('/tmp/imlabel.png')

    # computing normals (this will be needed for some features in the long run)
    # sc.im.normals = ne.compute_bilateral_normals(sc.im.depth)

    # we can get the labelling, which has been got from the ground truth image
    labels = np.unique(im_label[~np.isnan(im_label)])
    region_features = {}

    for label in labels:
        # label 0 is probably the floor
        if label == 0:
            continue

        region_features[label] = \
            fe.compute_features(sc.im, labelling == label)

    # # save the array of feature vector dicts to a file
    # D = dict(features=region_features, labelling=labelling)
    # # scipy.io.savemat(save_location, D)
    # with open(save_location, 'w') as f:
    #     pickle.dump(D, f, protocol=pickle.HIGHEST_PROTOCOL)


# need to import these *after* the pool helper has been defined
if False:
    # system_setup.multicore:
    import multiprocessing
    mapper = multiprocessing.Pool(system_setup.cores).map
else:
    mapper = map


if __name__ == '__main__':

    tic = time()
    mapper(process_sequence, paths.all_train_data[20:])
    print "In total took %f s" % (time() - tic)
