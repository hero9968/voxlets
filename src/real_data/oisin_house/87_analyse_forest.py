'''
Script to analyse the saved forest and print stats about it
Here I could do things like OOB error, importance etc.
'''
import sys
import os
import matplotlib.pyplot as plt
import yaml
import cPickle as pickle
import numpy as np

sys.path.append(os.path.expanduser('~/projects/shape_sharing/src/'))
from common import voxlets

import real_data_paths as paths

parameters_path = './training_params.yaml'
parameters = yaml.load(open(parameters_path))


def format_and_save_forest_histogram(H, savepath):
    plt.figure()
    plt.imshow(H, interpolation='nearest')
    plt.colorbar()
    plt.xlabel('Number of examples at leaf node')
    plt.ylabel('Depth of leaf node')
    plt.savefig(savepath)


def get_forest_stats(forest):
    '''
    Returns stats about the trees in the forest, e.g. their avg depth
    and so on...
    '''
    leaves_per_depth = np.zeros(forest.params['max_depth'])
    leaf_sizes = [[] for _ in range(forest.params['max_depth'])]
    for tree in forest.trees:
        leaves = tree.leaf_nodes()
        for leaf in leaves:
            this_depth = np.floor(np.log2(leaf.node_id+1)).astype(int)
            leaves_per_depth[this_depth] += 1
            leaf_sizes[this_depth].append(leaf.num_exs)

    width_to_plot = 30.0

    print ""
    print "\tHistogram over depths"
    print "\t====================="
    for depth, count in enumerate(leaves_per_depth):
        num_to_plot = int(count * (width_to_plot / leaves_per_depth.max()))
        string = "%6d:\t" + ("=" * num_to_plot) + ' %d'
        print string % (depth, count)
    print ""

    # generating a 2D histogram over the depth and size of the leaf nodes...
    max_leaf_size = 30
    H = np.zeros((len(leaves_per_depth), max_leaf_size))
    for depth, items in enumerate(leaf_sizes):
        for item in items:
            H[depth, min(item, max_leaf_size-1)] += 1

    return H


if __name__ == '__main__':

    # Repeat for each type of voxlet in the parameters
    for voxlet_params in parameters['voxlets']:

        for feature in parameters['features']:

            loadpath = paths.voxlet_model_path % (voxlet_params['name'], feature)
            loadpath = loadpath.replace('.pkl', '_full.pkl')
            model = pickle.load(open(loadpath))

            H = get_forest_stats(model.forest)
            savefolder = paths.models_folder % voxlet_params['name']
            format_and_save_forest_histogram(H, savefolder + 'forest_histogram.png')