'''
classes for extracting voxlets from grids, and for reforming grids from
voxlets.
'''

import numpy as np
import cPickle as pickle
import sys
import os
import time
import shutil
import copy
import paths
import voxel_data
import random_forest_structured as srf
import features
from skimage import measure
import subprocess as sp
from sklearn.neighbors import NearestNeighbors
import mesh
import sklearn.metrics
import collections
import random
import scipy.misc
#import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d.art3d import Poly3DCollection

sys.path.append(os.path.expanduser(
    '~/projects/shape_sharing/src/rendered_scenes/visualisation'))
import voxel_utils as vu

def render_diff_view(grid1, grid2, savepath):
    '''
    renders grid1, and any nodes not in grid2 get done in a different colour
    '''
    # convert nans to the minimum
    ms1 = mesh.Mesh()
    ms1.from_volume(grid1, 0)
    ms1.remove_nan_vertices()

    ms2 = mesh.Mesh()
    ms2.from_volume(grid2, 0)
    ms2.remove_nan_vertices()

    # now do a sort of setdiff between the two...
    print "Bulding dictionary", ms2.vertices.shape
    ms2_dict = {}
    for v in ms2.vertices:
        vt = (100*v).astype(int)
        ms2_dict[(vt[0], vt[1], vt[2])] = 1

    print "Done bulding dictionary", ms1.vertices.shape

    # label each vertex in ms1
    labels = np.zeros(ms1.vertices.shape[0])
    for count, v in enumerate(ms1.vertices):
        vt = (100*v).astype(int)
        if (vt[0], vt[1], vt[2]) in ms2_dict:
            labels[count] = 1
    print "Done checking dictionary"

    # memory problem?
    ms1.write_to_ply('/tmp/temp.ply', labels)

    sp.call([paths.blender_path,
             "../rendered_scenes/spinaround/spin.blend",
             "-b", "-P",
             "../rendered_scenes/spinaround/blender_spinaround_frame_ply.py"],
             stdout=open(os.devnull, 'w'),
             close_fds=True)

    #now copy file from /tmp/.png to the savepath...
    print "Moving render to " + savepath
    shutil.move('/tmp/.png', savepath)




def plot_mesh(verts, faces, ax):
    mesh = Poly3DCollection(verts[faces])
    mesh.set_alpha(0.8)
    mesh.set_edgecolor((1.0, 0.5, 0.5))
    ax.add_collection3d(mesh)

    ax.set_aspect('equal')
    MAX = 20
    for direction in (0, 1):
        for point in np.diag(direction * MAX * np.array([1,1,1])):
            ax.plot([point[0]], [point[1]], [point[2]], 'w')
    ax.axis('off')


def render_single_voxlet(V, savepath, level=0, short=False):

    assert V.ndim == 3
    print V.min(), V.max(), V.shape

    # renders a voxlet using the .blend file...
    temp = V.copy()
    # print savepath
    # print "Minmax is ", temp.min(), temp.max()

    # V[:, :, :] = np.nanmax(V)
    # add a bit around the edge, and adjust origin accordingly...
    crap = ((100.0, 100.0), (100.0, 100.0), (100.0, 100.0))
    temp = np.pad(temp, pad_width=1, mode='constant', constant_values=crap)

    verts, faces = measure.marching_cubes(temp, level)
    verts -= 0.005

    if np.any(np.isnan(verts)):
        import pdb; pdb.set_trace()

    D = dict(verts=verts, faces=faces)
    with open('/tmp/vertsfaces.pkl', 'wb') as f:
        pickle.dump(D, f)

    verts *= 0.0175 # parameters.Voxlet.size << bit of a magic number here...
    verts *= 10.0  # so its a reasonable scale for blender
    # print verts.min(axis=0), verts.max(axis=0)
    D = dict(verts=verts, faces=faces)
    with open('/tmp/vertsfaces.pkl', 'wb') as f:
        pickle.dump(D, f)
    vu.write_obj(verts, faces, '/tmp/temp_voxlet.obj')

    if short:
        blender_filepath = paths.RenderedData.voxlet_render_blend.replace('.blend', '_short.blend')
        print blender_filepath
    else:
        blender_filepath = paths.RenderedData.voxlet_render_blend

    sp.call([paths.blender_path,
         blender_filepath,
         "-b", "-P",
         paths.RenderedData.voxlet_render_script],
         stdout=open(os.devnull, 'w'),
         close_fds=True)

    #now copy file from /tmp/.png to the savepath...
    folderpath = os.path.split(savepath)[0]
    if not os.path.exists(folderpath):
        os.makedirs(folderpath)

    print "Moving render to " + savepath
    shutil.move('/tmp/temp_voxlet.png', savepath)

def voxlet_class_to_dict(params_class):
    voxlet_params = {}
    voxlet_params['tall_voxlets'] = params_class.tall_voxlets
    voxlet_params['one_side_bins'] = params_class.one_side_bins
    voxlet_params['shape'] = params_class.shape
    voxlet_params['size'] = params_class.size
    voxlet_params['actual_size'] = params_class.actual_size
    voxlet_params['centre'] = params_class.centre
    if hasattr(params_class, 'tall_voxlet_height'):
        voxlet_params['tall_voxlet_height'] = params_class.tall_voxlet_height
    return voxlet_params


class VoxletPredictor(object):
    '''
    Class to predict a full ixjxk voxlet given a feature vector
    Wraps a forest plus a PCA representation of the voxlets themselves.
    The computation of the PCA object is not done here as training voxlets are
    compressed with PCA as soon as they are extracted. This means that this
    class only ever sees the compressed versions.

    COULD make it so all the pca is done inside here - i.e. you keep giving it
    full voxlets before then training the model. Unsure. This would require
    saving the full voxlet version, if I were to keep to the same script
    ordering I have now...
    '''
    def __init__(self):
        pass

    def set_voxlet_params(self, voxlet_params):
        self.voxlet_params = voxlet_class_to_dict(voxlet_params)

    def set_pca(self, pca_in):
        self.pca = pca_in

    def set_feature_pca(self, feature_pca_in):
        self.feature_pca = feature_pca_in

    def set_masks_pca(self, masks_pca_in):
        self.masks_pca = masks_pca_in

    def train(self, X, Y, subsample_length=-1, masks=None, scene_ids=None):
        '''
        Runs the OMA forest code
        Y is expected to be a PCA version of the shoeboxes
        subsample_length is the maximum number of training examples to use.
        When it is -1, then we use all the training examples
        masks
            is an optional argument, giving a pca-ed binary mask for each
            training example
        '''
        if X.shape[0] != Y.shape[0]:
            raise Exception("X and Y should have the same number of rows")

        print "Before removing nans"
        self._print_shapes(X, Y)

        X, Y = self._remove_nans(X, Y)

        if subsample_length > 0 and subsample_length < X.shape[0]:
            X, Y, scene_ids = self._subsample(X, Y, scene_ids, subsample_length)

        print "After subsampling and removing nans...", subsample_length
        self._print_shapes(X, Y)

        print "Training forest"
        forest_params = srf.ForestParams()
        self.forest = srf.Forest(forest_params)
        tic = time.time()
        self.forest.train(X, Y, scene_ids)
        toc = time.time()
        print "Time to train forest is", toc-tic

        # must save the training data in this class, as the forest only saves
        # an index into the training set...
        self.training_Y = Y.astype(np.float16)
        self.training_X = X.astype(np.float16)

        # Unpythonic comparison but nessary in case it is numpy array
        if masks != None:
            self.training_masks = masks

    def _medioid_idx(self, data):
        '''
        similar to numpy 'mean', but returns the medioid data item
        '''
        # finding the distance to the mean
        mu = data.mean(axis=0)
        mu_dist = np.sqrt(((data - mu[np.newaxis, ...])**2).sum(axis=1))

        median_item_idx = mu_dist.argmin()
        return median_item_idx

    def reset_voxlet_counts(self):
        '''
        Reset the counter which counts how many times each training example is used
        '''
        self.voxlet_counter = np.zeros(self.training_Y.shape[0])

    def predict(self, X, how_to_choose='medioid', distance_measure='just_empty', visible_voxlet=None):
        '''
        Returns a voxlet prediction for a single X
        '''
        # each tree predicts which index in the test set to use...
        # rows = test data (X), cols = tree

        print X.shape
        index_predictions = self.forest.test(X).astype(int)
        self._cached_predictions = index_predictions
        # must extract original test data from the indices

        if how_to_choose == 'medioid':

            # this is a horrible line and needs changing...
            medioid_idx = [pred[self._medioid_idx(self.training_Y[pred])]
                                 for pred in index_predictions]
            Y_pred_compressed = [self.training_Y[idx] for idx in medioid_idx]
            Y_pred_compressed = np.array(Y_pred_compressed)

            all_masks = []
            #for each prediction...
            for row in index_predictions:
                # Want to get the mean mask for each data point in the leaf node...
                # Each row is a list of training examples
                these_masks_compressed = [self.training_masks[idx] for idx in row]
                # Must inverse transform *before* taking the mean -
                # I don't want to take mean in PCA space
                these_masks_full = \
                    self.masks_pca.inverse_transform(np.array(these_masks_compressed))
                all_masks.append(np.mean(these_masks_full, axis=0))
            all_masks = np.vstack(all_masks)

            final_predictions = self.pca.inverse_transform(Y_pred_compressed)

        elif how_to_choose == 'closest':
            # makes the prediction which is closest to the observed data...
            # candidates = sle
            assert X.shape[0] == 1 or len(X.shape) == 0

            X = X.flatten()
            visible_voxlet.flatten()

            index_predictions = index_predictions[0]
            assert len(index_predictions) > 5 and len(index_predictions) < 1000 # should be one prediction per tree

            tree_predictions = \
                self.pca.inverse_transform(self.training_Y[index_predictions])
            # print self.training_Y[index_predictions].shape
            # print "Tree predictions has shape ", tree_predictions.shape

            # the 0.9 is in case the PCA etc makes the nanmin not exacrtly correct

            # now we must be careful - use the biggest of the overlaps
            # between the predicted and the visible
            all_dims = []
            distances = []

            if distance_measure == 'largest_of_free_zones':
                dims_to_use_for_distance1 = \
                    visible_voxlet.flatten() > np.nanmin(visible_voxlet) * 0.9

                # print "Nan count in visible: ", np.isnan(visible_voxlet).sum()

                for tree_prediction in tree_predictions:
                    dims_to_use_for_distance2 = \
                        tree_prediction.flatten() > np.nanmin(visible_voxlet) * 0.9
                    # see which zone is bigger...
                    # print dims_to_use_for_distance1.sum(), dims_to_use_for_distance2.sum()
                    if dims_to_use_for_distance1.sum() > dims_to_use_for_distance2.sum():
                        # print "1 WINS"
                        dims_to_use_for_distance = np.copy(dims_to_use_for_distance1)
                    else:
                        # print "2 WINS"
                        dims_to_use_for_distance = np.copy(dims_to_use_for_distance2)

                    all_dims.append(dims_to_use_for_distance)
                    vec_dist = np.linalg.norm(
                        visible_voxlet.flatten()[dims_to_use_for_distance] -
                        tree_prediction[dims_to_use_for_distance])

                    distances.append(vec_dist / float(dims_to_use_for_distance.sum()))

                distances = np.array(distances)
                self._distances_cache = distances
                to_use = distances.argmin()
                self.min_dist = distances[to_use]
                self.dims_to_use_for_distance_cache = all_dims[to_use]

                final_predictions = tree_predictions[to_use]

            elif distance_measure == 'narrow_band':
                # now use the narrow band only...
                narrow_band = np.logical_and(
                    visible_voxlet.flatten() > np.nanmin(visible_voxlet) * 0.9,
                    visible_voxlet.flatten() < np.nanmax(visible_voxlet) * 0.9)

                narrow_band_distances = []

                mu = np.nanmax(tree_predictions)

                # narrow_band_wiggles = []
                for tree_prediction in tree_predictions:
                    # doing some wiggling...

                    # scales = [-mu/2, 0, mu/2]
                    # temp_dists = [np.linalg.norm(
                    #     visible_voxlet.flatten()[narrow_band] -
                    #     self._tsdf_scale(tree_prediction[narrow_band], scale, mu)) / \
                    #     float(narrow_band.sum())
                    #     for scale in scales]

                    SE = (visible_voxlet.flatten()[narrow_band] - tree_prediction[narrow_band])**2
                    MSE = np.mean(SE)
                    RMSE = np.sqrt(MSE)
                    narrow_band_distances.append(RMSE)

                    # minimum_distance_under_wiggle_idx = np.argmin(np.array(temp_dists))
                    # narrow_band_wiggles.append(scales[minimum_distance_under_wiggle_idx])

                # print tree_predictions[:, dims_to_use_for_distance].shape
                # print visible_voxlet.flatten()[dims_to_use_for_distance].shape
                distances = np.array(narrow_band_distances)
                # print "dists are ", distances
                # print "dists are ", distances
                self._distances_cache = distances

                # print "Distances has shape ", distances.shape

                to_use = distances.argmin()
                self.min_dist = distances[to_use]
                # this is nasty but hopefully ok...
                self.dims_to_use_for_distance_cache = narrow_band

                final_predictions = tree_predictions[to_use]

                # update the counter which counts how many times each voxlet is used
                if hasattr(self, 'voxlet_counter'):
                    self.voxlet_counter[index_predictions[to_use]] += 1
                    # self._tsdf_scale(
                    # narrow_band_wiggles[to_use], mu)

            elif distance_measure == 'just_empty':
                # This is the original measure I used to use, just looking at the
                # voxels known to be empty...
                dims_to_use_for_distance = \
                    visible_voxlet.flatten() > np.nanmin(visible_voxlet) * 0.9
                distances = np.linalg.norm(
                    visible_voxlet.flatten()[dims_to_use_for_distance] -
                    tree_predictions[:, dims_to_use_for_distance], axis=1)

                to_use = distances.argmin()
                self.min_dist = distances[to_use]
                # this is nasty but hopefully ok...
                self.dims_to_use_for_distance_cache = dims_to_use_for_distance
                self._distances_cache = distances

                final_predictions = tree_predictions[to_use]

            elif distance_measure == 'medioid':

                to_use = self._medioid_idx(tree_predictions)
                final_predictions = tree_predictions[to_use]

            elif distance_measure == 'mean':

                final_predictions = np.mean(tree_predictions, axis=0)

            # print "Retrieving this one...", index_predictions[to_use]

            if distance_measure == 'mean':

                these_masks_compressed = [self.training_masks[idx] for idx in index_predictions]
                # Must inverse transform *before* taking the mean -
                # I don't want to take mean in PCA space
                these_masks_full = \
                    self.masks_pca.inverse_transform(np.array(these_masks_compressed))

                all_masks = np.mean(these_masks_full, axis=0)
            else:
                if hasattr(self, 'training_masks'):
                    compressed_mask = self.training_masks[index_predictions[to_use]]
                    all_masks = np.vstack([self.masks_pca.inverse_transform(compressed_mask)])
                else:
                    all_masks = (final_predictions * 0) + 1
        else:
            raise Exception('Do not understand')

        return (final_predictions, all_masks)

    # def _medioid_idx(self, X):
    #     mu = X.mean(0)
    #     mu_dist = np.sqrt(((X - mu[np.newaxis, ...])**2).sum(1))
    #     return mu_dist.argmin()

    def _tsdf_scale(self, V, scale, mu):
        # apply some wiggle to the scale factor on the tsdf
        final_V = V + scale
        final_V[final_V>mu] = mu
        final_V[final_V<-mu] = -mu
        return final_V

    def save(self, savepath):
        '''
        Saves the model to specified file.
        I'm doing this as a method of the class so I can do the appropriate
        checks, as performed below
        '''
        if not hasattr(self, 'pca'):
            raise Exception(
                "pca attribute not set - this is important for prediction")

        if not hasattr(self, 'forest'):
            raise Exception("Forest not trained it seems")
        tic = time.time()

        with open(savepath, 'wb') as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)

        toc = time.time()
        print "Time to save forest is", toc-tic

    def _remove_nans(self, X, Y):
        '''
        Removes training entries with nans in feature space
        '''
        to_remove = np.any(np.isnan(X), axis=1)
        X = X[~to_remove, :]
        Y = Y[~to_remove, :]
        return X, Y

    def _subsample(self, X, Y, scene_ids, subsample_length):

        rand_exs = np.sort(np.random.choice(
            X.shape[0],
            np.minimum(subsample_length, X.shape[0]),
            replace=False))
        return X.take(rand_exs, 0), Y.take(rand_exs, 0), scene_ids.take(rand_exs, 0)

    def _print_shapes(self, X, Y):
        print "X has shape ", X.shape
        print "Y has shape ", Y.shape

    def render_leaf_nodes(self, folder_path, max_per_leaf=10, tree_id=0):
        '''
        renders all the voxlets at leaf nodes of a tree to a folder
        '''
        leaf_nodes = self.forest.trees[tree_id].leaf_nodes()

        print len(leaf_nodes)

        print "\n Sum of all leaf nodes is:"
        print sum([node.num_exs for node in leaf_nodes])

        print self.training_Y.shape

        if not os.path.exists(folder_path):
            raise Exception("Could not find path %s" % folder_path)

        print "Leaf node shapes are:"
        for node in leaf_nodes:
            print node.node_id, '\t', node.num_exs
            leaf_folder_path = folder_path + '/' + str(node.node_id) + '/'

            if not os.path.exists(leaf_folder_path):
                print "Creating folder %s" % leaf_folder_path
                os.makedirs(leaf_folder_path)

            if len(node.exs_at_node) > max_per_leaf:
                ids_to_render = random.sample(node.exs_at_node, max_per_leaf)
            else:
                ids_to_render = node.exs_at_node

            # Rendering each example at this node
            for count, example_id in enumerate(ids_to_render):
                V = self.pca.inverse_transform(self.training_Y[example_id])
                render_single_voxlet(V.reshape(self.voxlet_params['shape']),
                    leaf_folder_path + str(count) + '.png')

            with open(leaf_folder_path + ('total_%d.txt' % len(node.exs_at_node))) as f:
                f.write(len(node.exs_at_node))

            # Now doing the average mask and plotting slices through it
            mean_mask = self._get_mean_mask(ids_to_render)
            plt.figure(figsize=(10, 10))
            for count, slice_id in enumerate(range(0, mean_mask.shape[2], 10)):
                if count+1 > 3*3: break
                plt.subplot(3, 3, count+1)
                plt.imshow(mean_mask[:, :, slice_id], interpolation='nearest', cmap=plt.cm.gray)
                plt.clim(0, 1)
                plt.title('Slice_id = %d' % slice_id)
                plt.savefig(leaf_folder_path + 'slices.pdf')
                plt.close()

    def _get_mean_mask(self, training_idxs):
        '''
        returns the mean mask, given a set of indices into the training
        data
        '''
        all_masks = [self.masks_pca.inverse_transform(self.training_masks[idx])
                     for idx in training_idxs]
        return np.array(all_masks).mean(axis=0).reshape(self.voxlet_params['shape'])




class Reconstructer(object):
    '''
    does the final prediction
    '''

    def __init__(self, reconstruction_type, combine_type):
        self.reconstruction_type = reconstruction_type
        self.combine_type = combine_type

    def set_model(self, model):
        self.model = model

        if not hasattr(self.model, '__iter__'):
            self.nbrs = NearestNeighbors(n_neighbors=1, algorithm='kd_tree').fit(self.model.training_Y)

    def set_scene(self, sc_in):
        self.sc = sc_in

    def sample_points(self, num_to_sample, sample_grid_size, additional_mask=None):
        '''
        sampling points from the test image
        '''

        if False:
            # perhaps add in some kind of thing to make sure that we don't take normals pointing the wrong way
            xyz = self.sc.im.get_world_xyz()

            good_normals = self.sc.im.normals[:, 2].reshape((480, 640)) < -0.5
            if additional_mask:
                additional_mask = np.logical_and(additional_mask, good_normals)
            else:
                additional_mask = good_normals

            # Over-sample the points to begin with
            sampled_idxs = \
                self.sc.im.random_sample_from_mask(4*num_to_sample, additional_mask=additional_mask)
            linear_idxs = sampled_idxs[:, 0] * self.sc.im.mask.shape[1] + \
                sampled_idxs[:, 1]

            # Now I want to choose points according to the x-y grid
            # First, discretise the location of each point
            sampled_xy = xyz[linear_idxs, :2]
            sampled_xy /= sample_grid_size
            sampled_xy = np.round(sampled_xy).astype(int)

            sample_dict = {}
            for idx, row in zip(sampled_idxs, sampled_xy):
                if (row[0], row[1]) in sample_dict:
                    sample_dict[(row[0], row[1])].append(idx)
                else:
                    sample_dict[(row[0], row[1])] = [idx]

            sampled_points = []
            while len(sampled_points) < num_to_sample:

                for key, value in sample_dict.iteritems():
                    # add the top item to the sampled points
                    if len(value) > 0 and len(sampled_points) < num_to_sample:
                        sampled_points.append(value.pop())

            self.sampled_idxs = np.array(sampled_points)
        else:
            sample_rate = copy.copy(self.sc.im.depth)

            shape = self.sc.im.depth.shape
            sample_rate[self.sc.im.get_world_xyz()[:, 2].reshape(shape) < 0.035] = 0
            sample_rate[self.sc.im.mask==0] = 0
            sample_rate[self.sc.im.normals[:, 2].reshape(shape) > -0.1] = 0
            sample_rate[self.sc.im.get_world_normals()[:, 2].reshape(shape) > 0.98] = 0

            sample_rate /= sample_rate.sum()

            samples = np.random.choice(shape[0]*shape[1], num_to_sample, p=sample_rate.flatten())
            self.sampled_idxs = np.array(np.unravel_index(samples, shape)).T



    def _initialise_voxlet(self, index, model):
        '''
        given a point in an image, creates a new voxlet at an appropriate
        position and rotation in world space
        '''
        assert(index.shape[0] == 2)

        # getting the xyz and normals in world space
        world_xyz = self.sc.im.get_world_xyz()
        world_norms = self.sc.im.get_world_normals()

        # convert to linear idx
        point_idx = index[0] * self.sc.im.mask.shape[1] + index[1]

        # creating the voxlett10
        shoebox = voxel_data.ShoeBox(model.voxlet_params['shape'])  # grid size

        # creating the voxlet
        shoebox = voxel_data.ShoeBox(model.voxlet_params['shape'], np.float32)  # grid size
        shoebox.set_p_from_grid_origin(model.voxlet_params['centre'])  # m
        shoebox.set_voxel_size(model.voxlet_params['size'])  # m
        shoebox.V *= 0
        shoebox.V += self.sc.mu  # set the outside area to mu

        start_x = world_xyz[point_idx, 0]
        start_y = world_xyz[point_idx, 1]

        if model.voxlet_params['tall_voxlets']:
            start_z = model.voxlet_params['tall_voxlet_height']
        else:
            start_z = world_xyz[point_idx, 2]

        shoebox.initialise_from_point_and_normal(
            np.array([start_x, start_y, start_z]),
            world_norms[point_idx],
            np.array([0, 0, 1]))

        return shoebox

    def initialise_output_grid(self, gt_grid=None):
        '''defaulting to initialising from the ground truth grid...'''
        self.accum = voxel_data.UprightAccumulator(gt_grid.V.shape)
        self.accum.set_origin(gt_grid.origin, gt_grid.R
             )
        self.accum.set_voxel_size(gt_grid.vox_size)

        # during testing it makes sense to save the GT grid, for visualisation
        self.gt_grid = gt_grid

    def set_probability_model_one(self, ini):
        self.prob_model_one = ini

    def fill_in_output_grid_oma(self, render_type=[], render_savepath=None,
            use_implicit=False, oracle=None, add_ground_plane=False,
            combine_segments_separately=False, accum_only_predict_true=False,
            feature_collapse_type=None, feature_collapse_param=None,
            weight_empty_lower=None, use_binary=None, how_to_choose='closest',
            distance_measure='narrow_band', cobweb=False):
        '''
        Doing the final reconstruction
        In future, for this method could not use the image at all, but instead
        could make it so that the points and the normals are extracted directly
        from the tsdf

        oracle
            an optional argument which uses an oracle to choose the voxlets
            Choices of oracle should perhaps be in ['pca', 'ground_truth', 'nn' etc...]

        combine_segments_separately
            if true, then each segment in the image
            has a separate accumulation grid. These are then 'or'ed together
            (or similar) at the very end...

        accum_only_predict_true
            if true, the accumulator(s) will combine
            each prediction by only averaging in the regions predicted to be
            full. If false, the entire voxlet region will be added in, just
            like the CVPR submission.

        weight_empty_lower
            if set to a number, then empty regions will be weighted by this much.
            e.g. if set to 0.5, then empty regions will be weighted by 0.5.
            full regions are always weighted as 1.0

        use_binary:
            converts tsdf to binary before prediction and marging.
            MUST use only with a forest trained on binary predictions...
        '''

        if cobweb:
            cobwebengine = features.CobwebEngine(0.01, mask=self.sc.im.mask)
            cobwebengine.set_image(self.sc.im)
            self.cobwebengine=cobwebengine

        # saving
        # with open('/tmp/grid.pkl', 'wb') as f:
        #     pickle.dump(self.sc.gt_tsdf, f)

        # with open('/tmp/im.pkl', 'wb') as f:
        #     pickle.dump(self.sc.im, f)

        if combine_segments_separately:
            # Create a separate accumulator for each segment in the image...
            self.segement_accums = {}
            labels = np.unique(self.sc.visible_im_label[~np.isnan(self.sc.visible_im_label)])
            for label in labels:
                self.segement_accums[label] = self.accum.copy()
            self.accum = None

        self.all_pred_cache = []

        # if oracle == 'true_greedy' or oracle == 'true_greedy_gt':
        self.possible_predictions = []


        self.empty_voxels_in_voxlet_count = []
        self.gt_minus_predictions = []

        "extract features from each shoebox..."
        for count, idx in enumerate(self.sampled_idxs):

            if hasattr(self.model, '__iter__'):
                # randomly choose model to use...
                if np.random.rand() > self.prob_model_one:
                    model_to_use = self.model[1]
                else:
                    model_to_use = self.model[0]
            else:
                model_to_use = self.model

            tic = time.time()
            # find the segment index of this voxlet
            # this_point_label = self.sc.visible_im_label[idx[0], idx[1]]
            # get the voxel grid of tsdf associated with this label
            # BUT at test time how to get this segmented grid? We need a similar type thing to before...
            # this_idx_grid = self.sc.visible_tsdf_separate[this_point_label]
            this_idx_grid = self.sc.im_tsdf
            # print "WARNING - just using a single grid for the features..."
            # this_idx_grid = self.sc.im_tsdf

            "extract features from the tsdf volume"
            features_voxlet = self._initialise_voxlet(idx, model_to_use)
            features_voxlet.fill_from_grid(this_idx_grid)
            features_voxlet.V[np.isnan(features_voxlet.V)] = -self.sc.mu
            self.cached_feature_voxlet = features_voxlet.V

            if use_binary:
                features_voxlet.V = (features_voxlet.V > 0).astype(np.float16)

            if cobweb:
                feature_vector = cobwebengine.get_cobweb(idx)
                feature_vector[np.isnan(feature_vector)] = -5.0
                # print feature_vector
            else:
                feature_vector = self._feature_collapse(features_voxlet.V.flatten(),
                    feature_collapse_type, feature_collapse_param)

            # getting the GT voxlet - useful for the oracles and rendering
            gt_voxlet = self._initialise_voxlet(idx, model_to_use)
            gt_voxlet.fill_from_grid(self.sc.gt_tsdf)

            "Replace the prediction - if an oracle has been specified!"
            if oracle == 'gt':
                voxlet_prediction = gt_voxlet.V.flatten()
                weights_to_use = voxlet_prediction * 0 + 1

            elif oracle == 'pca':
                temp = model_to_use.pca.transform(gt_voxlet.V.flatten())
                voxlet_prediction = model_to_use.pca.inverse_transform(temp)
                weights_to_use = voxlet_prediction * 0 + 1

            elif oracle == 'nn':
                # getting the closest match in the training data...
                _, indices = self.nbrs.kneighbors(
                    model_to_use.pca.transform(gt_voxlet.V.flatten()))
                closest_training_Y = model_to_use.pca.inverse_transform(
                    model_to_use.training_Y[indices[0], :])
                voxlet_prediction = closest_training_Y
                weights_to_use = voxlet_prediction * 0 + 1

            else:
                # Doing a real prediction!

                A += time.time() - tic; tic = time.time()
                # print "time A :", A
                "classify according to the forest"
                voxlet_prediction, mask = \
                    model_to_use.predict(np.atleast_2d(feature_vector), how_to_choose=how_to_choose,
                        distance_measure=distance_measure, visible_voxlet=features_voxlet.V)
                self.cached_voxlet_prediction = voxlet_prediction
                # self.all_pred_cache.append(voxlet_prediction)

                self.empty_voxels_in_voxlet_count.append(
                    (voxlet_prediction > 0).sum().astype(float) / float(voxlet_prediction.size))
                self.gt_minus_predictions.append(
                    np.linalg.norm(voxlet_prediction - gt_voxlet.V.flatten()))

                # flipping the mask direction here:
                weights_to_use = 1-mask
                weights_to_use = weights_to_use.flatten()
                if weight_empty_lower:
                    # print "Weights has shape, ", weights_to_use.shape
                    # print "Voxlet prediction has shape ", voxlet_prediction.shape
                    weights_to_use[voxlet_prediction > 0] *= weight_empty_lower

            # adding the shoebox into the result
            transformed_voxlet = self._initialise_voxlet(idx, model_to_use)
            transformed_voxlet.V = voxlet_prediction.reshape(
                model_to_use.voxlet_params['shape'])

            if oracle == 'greedy_add':
                if combine_segments_separately:
                    acc_copy = self.segement_accums[this_point_label]
                else:
                    acc_copy = self.accum.copy()
                acc_copy.add_voxlet(transformed_voxlet, accum_only_predict_true, weights=weights_to_use)

                to_evaluate_on = np.logical_or(
                    self.sc.im_tsdf.V < 0, np.isnan(self.sc.im_tsdf.V))

                # now compare the two scores...
                gt_binary = self.sc.gt_tsdf.V[to_evaluate_on] > 0
                gt_binary[np.isnan(gt_binary)] = -self.sc.mu

                pred_new = acc_copy.compute_average().V[to_evaluate_on]
                pred_new[np.isnan(pred_new)] = +self.sc.mu

                new_auc = sklearn.metrics.roc_auc_score(gt_binary, pred_new)

                pred_old = self.accum.compute_average().V[to_evaluate_on]
                pred_old[np.isnan(pred_old)] = +self.sc.mu

                old_auc = sklearn.metrics.roc_auc_score(gt_binary, pred_old)

                if new_auc > old_auc:
                    if combine_segments_separately:
                        self.segement_accums[this_point_label] = acc_copy
                    else:
                        self.accum = acc_copy
                    print "Accepting! Increase of %f" % (new_auc - old_auc)
                else:
                    print "Rejecting! Would have decresed by %f" % (old_auc - new_auc)

            elif oracle == 'true_greedy' or oracle == 'true_greedy_gt':
                # store up all the predictions, wait until the end to add them in
                Di = {}
                # The distance used depends on if we are comparing to the ground truth or the observed data...
                if oracle == 'true_greedy':
                    Di['distance'] = model_to_use.min_dist
                elif oracle == 'true_greedy_gt':
                    Di['distance'] = np.linalg.norm(
                        transformed_voxlet.V.flatten()[model_to_use.dims_to_use_for_distance_cache.flatten()] -
                        gt_voxlet.V.flatten()[model_to_use.dims_to_use_for_distance_cache.flatten()])

                Di['voxlet'] = transformed_voxlet
                Di['mask'] = mask
                Di['weights'] = weights_to_use

                # saving this voxlet's position on a floorplan
                world_ij_grid = self.sc.gt_tsdf.world_xy_meshgrid()
                _, valid = transformed_voxlet.world_to_idx(world_ij_grid, detect_out_of_range=True)
                Di['floorplan'] = copy.copy(self.sc.gt_tsdf.V[:, :, 0]) * 0
                Di['floorplan'][valid.reshape(Di['floorplan'].shape)] = 1

                self.possible_predictions.append(Di)
            else:
                # Standard method - adding voxlet in regardless
                if combine_segments_separately:
                    raise Exception('Why????')
                    self.segement_accums[this_point_label].add_voxlet(transformed_voxlet, accum_only_predict_true, weights=weights_to_use)
                else:
                    try:
                        self.accum.add_voxlet(transformed_voxlet,
                            accum_only_predict_true, weights=weights_to_use)
                    except:
                        import pdb; pdb.set_trace()

            if 'blender' in render_type and render_savepath:

                # create a path of where to save the rendering
                savepath = render_savepath + '/%03d_%s.png'

                # doing rendering of the extracted grid
                render_single_voxlet(features_voxlet.V,
                    savepath % (count, 'extracted'))

                # doing rendering of the predicted grid
                render_single_voxlet(transformed_voxlet.V,
                    savepath % (count, 'predicted'))

                # doing rendering of the ground truth grid
                gt_voxlet = self._initialise_voxlet(idx, model_to_use)
                gt_voxlet.fill_from_grid(self.sc.gt_tsdf)
                render_single_voxlet(gt_voxlet.V,
                    savepath % (count, 'gt'))

                # render the closest voxlet in all of the leaf nodes to the GT

                render_single_voxlet(best_voxlet_V,
                    savepath % (count, 'gt'))

            if 'slice' in render_type:
                self._render-sl
                '''Plotting slices'''
                # Here want to now save slices at the corect high
                # in the extracted and predicted voxlets
                sf_x, sf_y = 2, 2
                plt.subplots(sf_x, sf_y)
                plt.subplots_adjust(left=0, bottom=0, right=0.98, top=0.95, wspace=0.02, hspace=0.02)

                plt.subplot(sf_x, sf_y, 1)
                plt.imshow(features_voxlet.V.reshape(model_to_use.voxlet_params['shape'])[:, :, 15], interpolation='nearest')
                plt.axis('off')
                plt.clim((-self.sc.mu, self.sc.mu))
                plt.title('Features voxlet')

                plt.subplot(sf_x, sf_y, 2)
                plt.imshow(transformed_voxlet.V[:, :, 15], interpolation='nearest')
                plt.axis('off')
                plt.clim((-self.sc.mu, self.sc.mu))
                plt.title('Forest prediction')


                # getting the world space coords
                world_xyz = self.sc.im.get_world_xyz()
                world_norms = self.sc.im.get_world_normals()

                # convert to linear idx
                point_idx = idx[0] * self.sc.im.mask.shape[1] + idx[1]

                temp = self.sc.gt_tsdf.world_to_idx(
                    world_xyz[point_idx, None])[0][:2]
                t_norm = world_norms[point_idx, :2]
                t_norm /= np.linalg.norm(t_norm)

                '''PLOTTING THE VOX LOCATION ON THE GT'''
                plt.subplot(sf_x, sf_y, 3)
                # np.nanmean(self.sc.gt_tsdf.V, axis=2)
                plt.imshow(self.sc.gt_tsdf.V[:, :, 15])
                # , cmap=plt.get_cmap('Greys'))
                plt.hold(True)
                self._plot_voxlet(temp, t_norm)
                plt.axis('off')
                plt.ylim(0, self.sc.gt_tsdf.V.shape[0])
                plt.xlim(0, self.sc.gt_tsdf.V.shape[1])

                '''PLOTTING THE VOX LOCATION ON THE IM TSDF'''
                plt.subplot(sf_x, sf_y, 4)
                # np.nanmean(self.sc.gt_tsdf.V, axis=2)
                tempim = self.sc.im_tsdf.V[:, :, 15]
                tempim[np.isnan(tempim)] = 0
                plt.imshow(tempim)
                plt.hold(True)
                self._plot_voxlet(temp, t_norm)
                plt.axis('off')
                plt.ylim(0, self.sc.gt_tsdf.V.shape[0])
                plt.xlim(0, self.sc.gt_tsdf.V.shape[1])

                # check the folder exists...
                if not os.path.exists(render_savepath + '/slices/'):
                    os.makedirs(render_savepath + '/slices/')
                saving_to = render_savepath + ('/slices/slice_%06d.png' % count)
                plt.savefig(saving_to)
                plt.close()

            if 'tree_predictions' in render_type:
                sf_x, sf_y = 6, 6
                plt.subplots(sf_x, sf_y)
                plt.subplots_adjust(left=0, bottom=0, right=0.98, top=0.95, wspace=0.02, hspace=0.02)
                sorted_idxs = np.argsort(model_to_use._distances_cache)
                for counter, sorted_idx in enumerate(sorted_idxs):
                    plt.subplot(sf_x, sf_y, counter+1)
                    pred_idx = model_to_use._cached_predictions[0][sorted_idx]
                    pred = model_to_use.pca.inverse_transform(model_to_use.training_Y[pred_idx])
                    plt.imshow(pred.reshape(model_to_use.voxlet_params['shape'])[:, :, 15])
                    plt.clim((-self.sc.mu, self.sc.mu))
                    plt.axis('off')

                if not os.path.exists(render_savepath + '/predictions/'):
                    os.makedirs(render_savepath + '/predictions/')
                plt.savefig(render_savepath + ('/predictions/pred_%06d.png' % count))
                plt.close()

            if 'tree_predictions' in render_type and 'slice' in render_type:

                plt.figure(figsize=(20, 20))
                plt.subplots(1, 2)
                plt.subplots_adjust(left=0, bottom=0, right=0.98, top=0.95, wspace=0.02, hspace=0.02)
                plt.subplot(121)
                plt.imshow(scipy.misc.imread(render_savepath + ('/slices/slice_%06d.png' % count)))
                plt.axis('off')
                plt.subplot(122)
                plt.imshow(scipy.misc.imread(render_savepath + ('/predictions/pred_%06d.png' % count)))
                plt.axis('off')
                if not os.path.exists(render_savepath + '/combined_slices/'):
                    os.makedirs(render_savepath + '/combined_slices/')
                plt.savefig(render_savepath + ('/combined_slices/combined_%06d.png' % count))
                plt.close()

            if 'matplotlib' in render_type:

                '''matplotlib 3d plots in subfigs'''

                plt.clf()
                fig = plt.figure(1, figsize=(18, 18))

                '''create range of items'''
                vols = ((features_voxlet.V, 'Observed'),
                           (transformed_voxlet.V, 'Predicted'),
                           (gt_voxlet.V, 'gt'),
                           (closest_training_Y, 'closest training Y'))

                for vol_count, (V, title) in enumerate(vols):

                    verts, faces = measure.marching_cubes(V.reshape(model_to_use.voxlet_params['shape']), 0)

                    ax = fig.add_subplot(2, 2, vol_count+1, projection='3d', aspect='equal')
                    plot_mesh(verts, faces, ax)
                    plt.title(title)

                plt.tight_layout()
                savepath = render_savepath + '/compiled_%03d.png' % count
                plt.savefig(savepath)
                plt.close()

        if oracle == 'true_greedy' or oracle == 'true_greedy_gt':
            # in true greedy, then we wait until here to add everything together...
            # sort so the smallest possible predictions are at the front...

            # this is when to save the grid for visualisation purposes...
            stop_points = [10, 50, 100, 200]

            results = collections.OrderedDict()
            self.possible_predictions.sort(key=lambda x: x['distance'])

            # # create a floor plan of the union of all floor plans...
            # all_fplans = np.dstack([p['floorplan'] for p in self.possible_predictions])
            # self.union_fplan = np.any(all_fplans, axis=2)

            # now create a blank floorplan which keeps track of how much area has been filled...
            # filled_fplan = self.union_fplan.copy().astype(bool) * 0

            for count, prediction in enumerate(self.possible_predictions):

                # print "The potential new energies are ", self._eval_union_cost(
                #     self.union_fplan, filled_fplan, all_fplans)

                # add this one in...
                self.accum.add_voxlet(prediction['voxlet'],
                    accum_only_predict_true, weights=prediction['weights'])
                # filled_fplan = np.logical_or(filled_fplan, prediction['floorplan'])

                if count in stop_points:
                    results[count] = copy.deepcopy(self.accum.compute_average())
                    # good for memory...
                    results[count].sumV = []
                    results[count].countV = []

                # print "energies are ", self._eval_union_cost(
                #     self.union_fplan, filled_fplan, all_fplans)


            return results

        if combine_segments_separately:
            raise Exception('Why am I here???')
            with open('/tmp/separate.pkl', 'w') as f:
                pickle.dump(self.segement_accums, f, protocol=pickle.HIGHEST_PROTOCOL)
            average = self.sc.gt_tsdf.blank_copy()
            average.V += self.sc.mu
            for _, accum in self.segement_accums.iteritems():
                temp_average = accum.compute_average()
                print "This sum is ", temp_average.V.sum()
                to_use = temp_average.V < 0
                print "To use sum is ", to_use.sum()
                average.V[to_use] = temp_average.V[to_use]
        else:
            average = self.accum.compute_average()

        # creating a final output which preserves the existing geometry
        keeping_existing = self.sc.im_tsdf.copy()
        to_use_prediction = np.isnan(keeping_existing.V)
        keeping_existing.V[to_use_prediction] = \
            average.V[to_use_prediction]
        self.keeping_existing = keeping_existing

        if add_ground_plane:

            # Adding in the ground plane
            self.keeping_existing.V[:, :, :add_ground_plane] = -self.sc.mu
            self.keeping_existing.V[:, :, add_ground_plane] = self.sc.mu
            average.V[:, :, :add_ground_plane] = -self.sc.mu
            average.V[:, :, add_ground_plane] = self.sc.mu

        if use_binary:
            # making sure the level set is at zero!
            average.V += 0.5
            self.keeping_existing.V += 0.5

        # good for memory...
        average.sumV = []
        average.countV = []

        # removing the excess from the grid...
        self.remove_excess = average.copy()
        self.remove_excess.V[self.sc.im_tsdf.V > 0] = self.sc.mu

        return self.remove_excess

    def _blender_render(self, features_voxlet, transformed_voxlet, render_savepath):
            # create a path of where to save the rendering
            savepath = render_savepath + '/%03d_%s.png'

            # doing rendering of the extracted grid
            render_single_voxlet(features_voxlet.V,
                savepath % (count, 'extracted'))

            # doing rendering of the predicted grid
            render_single_voxlet(transformed_voxlet.V,
                savepath % (count, 'predicted'))

            # doing rendering of the ground truth grid
            gt_voxlet = self._initialise_voxlet(idx)
            gt_voxlet.fill_from_grid(self.sc.gt_tsdf)
            render_single_voxlet(gt_voxlet.V,
                savepath % (count, 'gt'))

            # render the closest voxlet in all of the leaf nodes to the GT
            render_single_voxlet(best_voxlet_V,
                savepath % (count, 'gt'))

    def save_voxlet_counts(self, fpath):
        # for each model...
        for model_idx, model in enumerate(self.models):
            # writes a count of how many of each voxlet was used
            with open(fpath + 'voxlet_count_%02d.txt' % idx, 'w'):
                for count in model.voxlet_counter:
                    f.write("%d, ", count)

    def save_empty_voxel_counts(self, fpath):
        # Save how many voxels are empty in each of the voxlet predictions
        with open(fpath + 'empty_voxel_count.txt', 'w') as f:
            for empty_ans in self.empty_voxels_in_voxlet_count:
                f.write('%0.5f\n' % empty_ans)

        # Do some kind of histogram plot
        plt.figure()
        plt.hist(np.array(self.empty_voxels_in_voxlet_count), 100)
        plt.xlim(0, 1.0)
        plt.xlabel('Fraction of voxels which are empty in the voxlet')
        plt.ylabel('Count')
        plt.savefig(fpath + 'empty_voxel_count.png')
        plt.close()

    def save_difference_from_predictions(self, fpath):
        # Save how many voxels are empty in each of the voxlet predictions
        with open(fpath + 'gt_minus_predictions.txt', 'w') as f:
            for empty_ans in self.gt_minus_predictions:
                f.write('%0.5f\n' % empty_ans)

        # Do some kind of histogram?
        plt.figure()
        plt.hist(np.array(self.gt_minus_predictions), 100)
        plt.xlabel('Ground truth minus the voxlet prediction')
        plt.ylabel('Count')
        plt.savefig(fpath + 'gt_minus_predictions.png')
        plt.close()


    def _feature_collapse(self, X, feature_collapse_type, parameter):
        """Applied to the feature shoeboxes after extraction"""

        if feature_collapse_type == 'pca':
            return self.model.feature_pca.transform(X.flatten())

        elif feature_collapse_type == 'decimate':
            X_sub = X[::parameter, ::parameter, ::parameter]
            return X_sub.flatten()

        else:
            raise Exception('Unknown feature collapse type')

    def plot_sampled_points(self, savepath=None):
        plt.imshow(self.sc.im.depth)
        plt.hold(True)
        plt.plot(self.sampled_idxs[:, 1], self.sampled_idxs[:, 0], 'ro')
        plt.hold(False)

    def plot_voxlet_top_view(self, savepath=None):
        '''
        plot the voxlets from the top
        '''
        fig = plt.figure(figsize=(25, 10), dpi=1000)
        plt.subplots(1, 3)
        plt.subplots_adjust(left=0, bottom=0, right=0.99, top=0.99, wspace=0, hspace=0)

        plt.subplot(131)
        plt.imshow(self.sc.im.rgb)
        plt.axis('off')
        plt.subplot(132)

        top_view = np.nanmean(self.sc.gt_tsdf.V, axis=2)
        # np.nanmean(self.sc.im_tsdf.V, axis=2)
        plt.imshow(top_view, cmap=plt.get_cmap('Greys'))
        plt.axis('off')

        plt.subplot(133)
        plt.imshow(top_view, cmap=plt.get_cmap('Greys'))
        plt.hold(True)

        # getting the world space coords
        world_xyz = self.sc.im.get_world_xyz()
        world_norms = self.sc.im.get_world_normals()

        for index in self.sampled_idxs:
            # convert to linear idx
            point_idx = index[0] * self.sc.im.mask.shape[1] + index[1]

            temp = self.sc.gt_tsdf.world_to_idx(
                world_xyz[point_idx, None])[0][:2]
            t_norm = world_norms[point_idx, :2]
            t_norm /= np.linalg.norm(t_norm)
            self._plot_voxlet(temp, t_norm)

        plt.axis('off')

        plt.subplot(133).set_xlim(plt.subplot(132).get_xlim())
        plt.subplot(133).set_ylim(plt.subplot(132).get_ylim())

        if savepath:
            plt.savefig(savepath.replace('png', 'pdf'), dpi=400)
            plt.close()

    def _get_voxlet_corners(self):

        if hasattr(self.model, '__iter__'):
            model_to_use = self.model[0]
        else:
            model_to_use = self.model

        v_size = model_to_use.voxlet_params['size'] * np.array(model_to_use.voxlet_params['shape'])
        cen = model_to_use.voxlet_params['centre']

        c1 = [-cen[0], cen[1]]
        c2 = [cen[0], cen[1]]
        c3 = [cen[0], cen[1] - v_size[1]]
        c4 = [-cen[0], cen[1] - v_size[1]]

        corners = np.array((c2, c1, c4, c3, c2)) / self.sc.gt_tsdf.vox_size
        return corners[:, ::-1]

    def _plot_voxlet(self, point, normal):
        plt.plot(point[1], point[0], 'or', ms=2)
        scaling = 10.0
        end_point = point + scaling * normal
        plt.plot([point[1], end_point[1]], [point[0], end_point[0]], 'r', lw=1)

        corners = self._get_voxlet_corners()

        norm2 = np.array([-normal[1], normal[0]])
        R = np.vstack((normal, norm2)).T

        t_corners = np.dot(R, corners.T).T + point
        plt.plot(t_corners[:, 1], t_corners[:, 0], '-g', lw=1)


class EnergySolver(object):
    '''
    strange class to solve the final energy function
    '''
    def __init__(self, prediction_dics, accum):
        # this is a dictionary of all the predictions
        blank_grid = accum.blank_copy()

        self.pred_dics = prediction_dics

        # seeing which cells would be full if *all* the cells were full
        temp = blank_grid.blank_copy()
        for p in self.possible_predictions:
            temp.add_voxlet(p['voxlet'])
        self.union_fplan = temp.countV > 0
        self.num_possible_voxels = np.sum(self.union_fplan)
        print "There are %d possible voxels " % self.num_possible_voxels

        # this is where we will fill all the predictions into
        self.accum = accum


    def solve(alpha, beta):
        pass
        # get the full f_plan

        # alpha beta are parameters...

        # get the current floorplan from the grid


    def _eval_union_cost(self, union_fplan, current_fplan, proposed_additions):
        '''
        evaluates the new value of the energy function under each of the
        proposed additions, which is a list of proposed additions...
        '''

        full_fplan = np.atleast_3d(union_fplan) * \
            (np.atleast_3d(current_fplan.astype(bool)) + proposed_additions.astype(bool))
        # print "full_fplan is ", full_fplan.shape, full_fplan.max(), full_fplan.min()
        sums = np.sum(np.sum(full_fplan, axis=0), axis=0)
        # print "energy is shape ", energy.shape
        energies = 1 - sums.astype(np.float32) / float(np.sum(union_fplan))
        # print "energies are ", energies.min(), energies.max()
        return energies






class VoxelGridCollection(object):
    '''
    class for doing things to a list of same-sized voxelgrids
    Not ready to use yet - but might be good one day!
    '''
    def __init__(self):
        raise Exception("Not ready to use!")

    def set_voxelgrids(self, voxgrids_in):
        self.voxgrids = voxgrids_in

    def cluster_voxlets(self, num_clusters, subsample_length):

        '''helper function to cluster voxlets'''

        # convert to np array
        np_all_sboxes = np.concatenate(shoeboxes, axis=0)
        all_sboxes = np.array([sbox.V.flatten() for sbox in self.voxlist]).astype(np.float16)

        # take subsample
        if local_subsample_length > X.shape[0]:
            X_subset = X
        else:
            to_use_for_clustering = \
                np.random.randint(0, X.shape[0], size=(local_subsample_length))
            X_subset = X[to_use_for_clustering, :]

        # doing clustering
        km = MiniBatchKMeans(n_clusters=num_clusters)
        km.fit(X_subset)
