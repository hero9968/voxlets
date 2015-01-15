'''
a script to set up the folders for the training data, also to do things like
the partial kinfu perhaps...
and I think now this should actually do the full thing to extract and save
training data from each sequence
'''

import sys
import os
import numpy as np
import yaml
import scipy.io

sys.path.append(os.path.expanduser('~/projects/shape_sharing/src/'))
sys.path.append(os.path.expanduser('~/projects/shape_sharing/src/intrinsic/'))
from common import paths
from common import images
from common import voxel_data
from common import carving
from features import line_casting


with open(paths.yaml_train_location, 'r') as f:
    train_sequences = yaml.load(f)

for sequence in train_sequences:

    seq_foldername = paths.sequences_save_location + sequence['name'] + '/'
    print "Creating %s" % seq_foldername
    if not os.path.exists(seq_foldername):
        os.makedirs(seq_foldername)

    # load the full video (in future possibly change this to be able to load
    # a subvideo...)
    input_data_path = paths.scenes_location + sequence['scene']
    vid = images.RGBDVideo()
    vid.load_from_yaml(input_data_path, 'poses.yaml')

    # load in the ground truth grid for this scene, and converting nans
    gt_vox = voxel_data.load_voxels(input_data_path + '/voxelgrid.pkl')
    gt_vox.V[np.isnan(gt_vox.V)] = -0.03

    # do the tsdf fusion from these frames
    # note the slightly strange partial_tsdf copy and overright
    # need the copy for the R, origin etc. Don't give gt to the carver
    # to preserve train/test integrity!
    carver = carving.Fusion()
    carver.set_video(vid.subvid(sequence['frames']))
    partial_tsdf = gt_vox.blank_copy()
    carver.set_voxel_grid(partial_tsdf)
    partial_tsdf = carver.fuse()

    # save the grid
    partial_tsdf.save(seq_foldername + 'input_fusion.pkl')

    # find the visible voxels ...? es ok
    vis = carving.VisibleVoxels()
    vis.set_voxel_grid(partial_tsdf)
    visible = vis.find_visible_voxels()

    # save the visible voxels
    visible.save(seq_foldername + 'visible_voxels.pkl')

    # getting the known full and empty voxels based on the depth image
    known_full_voxels = visible
    known_empty_voxels = partial_tsdf.blank_copy()
    known_empty_voxels.V = partial_tsdf.V > 0

    X, Y = line_casting.feature_pairs_3d(
        known_empty_voxels, known_full_voxels, gt_vox.V,
        samples=1000, base_height=0, autorotate=False)

    training_pairs = dict(X=X, Y=Y)
    scipy.io.savemat(seq_foldername + 'training_pairs.mat', training_pairs)

