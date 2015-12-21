'''
Collate the results to find the most popular voxlets
'''
import os, sys
sys.path.append(os.path.expanduser("~/projects/shape_sharing/src/"))
sys.path.append(os.path.expanduser("~/projects/shape_sharing/src/real_data/"))
import yaml
from oisin_house import real_data_paths as paths
import numpy as np

test_type = 'different_data_split'

for model_idx in [0, 1]:

    all_counts = []

    for sequence in paths.test_data:

        fpath = paths.voxlet_prediction_folderpath % \
            (test_type, sequence['name'])
        count_path = fpath + 'voxlet_counts.csv' + 'voxlet_count_%02d.txt' % model_idx

        voxlet_counts = np.genfromtxt(count_path, delimiter=',')
        all_counts.append(voxlet_counts)
    all_counts = np.vstack(all_counts)

    print "Now forming full array..."
    max_vox_id = all_counts[:, 0].max()
    accumulator = np.zeros(max_vox_id+1, int)
    for idx, val in all_counts:
        accumulator[idx] += val


    savepath = os.path.expanduser('./model_%02d_total.txt' % model_idx)

    np.savetxt(savepath, accumulator.astype(int), fmt="%d", delimiter=",")

    # do a plot here...
