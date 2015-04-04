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
import parameters
import voxel_data
import random_forest_structured as srf
import features
from skimage import measure
import subprocess as sp
from sklearn.neighbors import NearestNeighbors
import mesh
import sklearn.metrics

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


def render_single_voxlet(V, savepath, level=0):

    assert V.ndim == 3
    print V.min(), V.max(), V.shape

    # renders a voxlet using the .blend file...
    temp = V.copy()
    print savepath
    print "Minmax is ", temp.min(), temp.max()

    V[:, :, -2:] = parameters.RenderedVoxelGrid.mu
    verts, faces = measure.marching_cubes(V, level)

    if np.any(np.isnan(verts)):
        import pdb; pdb.set_trace()

    D = dict(verts=verts, faces=faces)
    with open('/tmp/vertsfaces.pkl', 'wb') as f:
        pickle.dump(D, f)

    verts *= parameters.Voxlet.size
    verts *= 10.0  # so its a reasonable scale for blender
    print verts.min(axis=0), verts.max(axis=0)
    D = dict(verts=verts, faces=faces)
    with open('/tmp/vertsfaces.pkl', 'wb') as f:
        pickle.dump(D, f)
    vu.write_obj(verts, faces, '/tmp/temp_voxlet.obj')

    sp.call([paths.blender_path,
         paths.RenderedData.voxlet_render_blend,
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
            X, Y = self._subsample(X, Y, subsample_length)

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
        self.training_Y = Y
        self.training_X = X

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

    def predict(self, X):
        '''
        Returns a voxlet prediction for each row in X
        '''
        # each tree predicts which index in the test set to use...
        # rows = test data (X), cols = tree
        index_predictions = self.forest.test(X).astype(int)
        print "Forest predicts ", index_predictions
        # must extract original test data from the indices

        # this is a horrible line and needs changing...
        medioid_idx = [self._medioid_idx(self.training_Y[pred])
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

        print "X has shape ", X.shape
        print "All masks has shape ", all_masks.shape

        return (self.pca.inverse_transform(Y_pred_compressed), all_masks)

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

    def _subsample(self, X, Y, subsample_length):

        rand_exs = np.sort(np.random.choice(
            X.shape[0],
            np.minimum(subsample_length, X.shape[0]),
            replace=False))
        return X.take(rand_exs, 0), Y.take(rand_exs, 0)

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
                ids_to_render = node.exs_at_node[:max_per_leaf]
            else:
                ids_to_render = node.exs_at_node

            # Rendering each example at this node
            for count, example_id in enumerate(ids_to_render):
                V = self.pca.inverse_transform(self.training_Y[example_id])
                render_single_voxlet(V.reshape(parameters.Voxlet.shape),
                    leaf_folder_path + str(count) + '.png')

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

    def _get_mean_mask(self, training_idxs):
        '''
        returns the mean mask, given a set of indices into the training
        data
        '''
        all_masks = [self.masks_pca.inverse_transform(self.training_masks[idx])
                     for idx in training_idxs]
        return np.array(all_masks).mean(axis=0).reshape(parameters.Voxlet.shape)




class Reconstructer(object):
    '''
    does the final prediction
    '''

    def __init__(self, reconstruction_type, combine_type):
        self.reconstruction_type = reconstruction_type
        self.combine_type = combine_type

    def set_model(self, model):
        self.model = model

        self.nbrs = NearestNeighbors(n_neighbors=1, algorithm='kd_tree').fit(self.model.training_Y)

    def set_scene(self, sc_in):
        self.sc = sc_in

    def sample_points(self, num_to_sample):
        '''
        sampling points from the test image
        '''
        sample_grid_size = parameters.VoxletPrediction.sampling_grid_size

        # Over-sample the points to begin with
        sampled_idxs = \
            self.sc.im.random_sample_from_mask(4*num_to_sample)
        linear_idxs = sampled_idxs[:, 0] * self.sc.im.mask.shape[1] + \
            sampled_idxs[:, 1]

        # Now I want to choose points according to the x-y grid
        # First, discretise the location of each point
        xyz = self.sc.im.get_world_xyz()
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
                if len(value) > 0:
                    sampled_points.append(value.pop())

        self.sampled_idxs = np.array(sampled_points)

    def _initialise_voxlet(self, index):
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
        shoebox = voxel_data.ShoeBox(parameters.Voxlet.shape)  # grid size

        # creating the voxlet
        shoebox = voxel_data.ShoeBox(parameters.Voxlet.shape, np.float32)  # grid size
        shoebox.set_p_from_grid_origin(parameters.Voxlet.centre)  # m
        shoebox.set_voxel_size(parameters.Voxlet.size)  # m
        shoebox.V *= 0
        shoebox.V += parameters.RenderedVoxelGrid.mu  # set the outside area to mu

        start_x = world_xyz[point_idx, 0]
        start_y = world_xyz[point_idx, 1]

        if parameters.Voxlet.tall_voxlets:
            start_z = parameters.Voxlet.tall_voxlet_height
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
        self.accum.set_origin(gt_grid.origin)
        self.accum.set_voxel_size(gt_grid.vox_size)

        # during testing it makes sense to save the GT grid, for visualisation
        self.gt_grid = gt_grid

    def fill_in_output_grid_oma(self, render_type, render_savepath=None,
            use_implicit=False, oracle=None, add_ground_plane=False,
            combine_segments_separately=False, accum_only_predict_true=False):
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
        '''

        # saving
        with open('/tmp/grid.pkl', 'wb') as f:
            pickle.dump(self.sc.gt_tsdf, f)

        with open('/tmp/im.pkl', 'wb') as f:
            pickle.dump(self.sc.im, f)

        if combine_segments_separately:
            # Create a separate accumulator for each segment in the image...
            self.segement_accums = {}
            labels = np.unique(self.sc.visible_im_label[~np.isnan(self.sc.visible_im_label)])
            for label in labels:
                self.segement_accums[label] = self.accum.copy()
            self.accum = None

        "extract features from each shoebox..."
        for count, idx in enumerate(self.sampled_idxs):

            # find the segment index of this voxlet
            this_point_label = self.sc.visible_im_label[idx[0], idx[1]]
            # get the voxel grid of tsdf associated with this label
            # BUT at test time how to get this segmented grid? We need a similar type thing to before...
            this_idx_grid = self.sc.visible_tsdf_separate[this_point_label]
            # print "WARNING - just using a single grid for the features..."
            # this_idx_grid = self.sc.im_tsdf

            # extract features from the tsdf volume
            features_voxlet = self._initialise_voxlet(idx)
            features_voxlet.fill_from_grid(this_idx_grid)
            features_voxlet.V[np.isnan(features_voxlet.V)] = \
                -parameters.RenderedVoxelGrid.mu
            self.cached_feature_voxlet = features_voxlet.V

            if use_implicit:
                implicit_voxlet = self._initialise_voxlet(idx)
                implicit_voxlet.fill_from_grid(self.sc.implicit_tsdf)
                self.cached_implicit_voxlet = implicit_voxlet.V

                combined_feature = np.concatenate(
                    (features_voxlet.V.flatten(),
                     implicit_voxlet.V.flatten()), axis=1)

            else:
                combined_feature = features_voxlet.V.flatten()

            feature_vector = self._feature_collapse(combined_feature)

            "classify according to the forest"
            voxlet_prediction, mask_prediction = \
                self.model.predict(np.atleast_2d(feature_vector))
            self.cached_voxlet_prediction = voxlet_prediction

            # getting the GT voxlet
            gt_voxlet = self._initialise_voxlet(idx)
            gt_voxlet.fill_from_grid(self.sc.gt_tsdf)

            # getting the closest match in the training data...
            ans, indices = self.nbrs.kneighbors(self.model.pca.transform(gt_voxlet.V.flatten()))
            closest_training_X = self.model.training_X[indices[0], :]
            closest_training_Y = self.model.pca.inverse_transform(
                self.model.training_Y[indices[0], :])

            "Replace the prediction - if an oracle has been specified!"
            if oracle == 'gt':
                voxlet_prediction = gt_voxlet.V.flatten()

            elif oracle == 'pca':
                temp = self.model.pca.transform(gt_voxlet.V.flatten())
                voxlet_prediction = self.model.pca.inverse_transform(temp)

            elif oracle == 'nn':
                voxlet_prediction = closest_training_Y

            # adding the shoebox into the result
            transformed_voxlet = self._initialise_voxlet(idx)
            transformed_voxlet.V = voxlet_prediction.reshape(
                parameters.Voxlet.shape)

            if oracle == 'greedy_add':
                if combine_segments_separately:
                    acc_copy = self.segement_accums[this_point_label]
                else:
                    acc_copy = self.accum.copy()
                acc_copy.add_voxlet(transformed_voxlet, accum_only_predict_true, weights=mask_prediction)

                to_evaluate_on = np.logical_or(
                    self.sc.im_tsdf.V < 0, np.isnan(self.sc.im_tsdf.V))

                # now compare the two scores...
                gt_binary = self.sc.gt_tsdf.V[to_evaluate_on] > 0
                gt_binary[np.isnan(gt_binary)] = -parameters.RenderedVoxelGrid.mu

                pred_new = acc_copy.compute_average().V[to_evaluate_on]
                pred_new[np.isnan(pred_new)] = +parameters.RenderedVoxelGrid.mu

                new_auc = sklearn.metrics.roc_auc_score(gt_binary, pred_new)

                pred_old = self.accum.compute_average().V[to_evaluate_on]
                pred_old[np.isnan(pred_old)] = +parameters.RenderedVoxelGrid.mu

                old_auc = sklearn.metrics.roc_auc_score(gt_binary, pred_old)

                if new_auc > old_auc:
                    if combine_segments_separately:
                        self.segement_accums[this_point_label] = acc_copy
                    else:
                        self.accum = acc_copy
                    print "Accepting! Increase of %f" % (new_auc - old_auc)
                else:
                    print "Rejecting! Would have decresed by %f" % (old_auc - new_auc)
            else:
                # Standard method - adding voxlet in regardless
                if combine_segments_separately:
                    self.segement_accums[this_point_label].add_voxlet(transformed_voxlet, accum_only_predict_true, weights=mask_prediction)
                else:
                    self.accum.add_voxlet(transformed_voxlet,
                        accum_only_predict_true, weights=mask_prediction)

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
                gt_voxlet = self._initialise_voxlet(idx)
                gt_voxlet.fill_from_grid(self.sc.gt_tsdf)
                render_single_voxlet(gt_voxlet.V,
                    savepath % (count, 'gt'))

                # render the closest voxlet in all of the leaf nodes to the GT

                render_single_voxlet(best_voxlet_V,
                    savepath % (count, 'gt'))

                mu = parameters.RenderedVoxelGrid.mu

            if 'slice' in render_type:
                '''Plotting slices'''
                # Here want to now save slices at the corect high
                # in the extracted and predicted voxlets
                sf_x, sf_y = 2, 2
                plt.subplot(sf_x, sf_y, 1)
                plt.imshow(feature_vector.reshape(parameters.Voxlet.shape[:2]), interpolation='nearest')
                plt.clim((-mu, mu))
                plt.title('Features voxlet')

                plt.subplot(sf_x, sf_y, 2)
                plt.imshow(transformed_voxlet.V[:, :, 15], interpolation='nearest')
                plt.clim((-mu, mu))
                plt.title('Forest prediction')

                plt.subplot(sf_x, sf_y, 3)
                # extracting the nearest training neighbour
                ans, indices = self.nbrs.kneighbors(feature_vector)
                NN = self.model.training_X[indices[0], :].reshape(parameters.Voxlet.shape[:2])
                plt.imshow(NN, interpolation='nearest')
                plt.clim((-mu, mu))
                plt.title('Nearest neighbour (X)')

                plt.subplot(sf_x, sf_y, 4)
                # extracting the nearest training neighbour

                NN_Y = self.model.pca.inverse_transform(self.model.training_Y[indices[0], :])
                NN_Y = NN_Y.reshape(parameters.Voxlet.shape)[:, :, 15]
                plt.imshow(NN_Y, interpolation='nearest')
                plt.clim((-mu, mu))
                plt.title('Nearest neighbour (Y)')

                plt.savefig(savepath % (count, 'slice'))

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

                    verts, faces = measure.marching_cubes(V.reshape(parameters.Voxlet.shape), 0)

                    ax = fig.add_subplot(2, 2, vol_count+1, projection='3d', aspect='equal')
                    plot_mesh(verts, faces, ax)
                    plt.title(title)

                plt.tight_layout()
                savepath = render_savepath + '/compiled_%03d.png' % count
                plt.savefig(savepath)

        if combine_segments_separately:
            with open('/tmp/separate.pkl', 'w') as f:
                pickle.dump(self.segement_accums, f, protocol=pickle.HIGHEST_PROTOCOL)
            average = self.sc.gt_tsdf.blank_copy()
            average.V += parameters.RenderedVoxelGrid.mu
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
        self.keeping_existing = keeping_existing
        keeping_existing.V[to_use_prediction] = \
            average.V[to_use_prediction]

        if add_ground_plane:
            # Adding in the ground plane
            self.keeping_existing.V[:, :, :4] = -parameters.RenderedVoxelGrid.mu
            self.keeping_existing.V[:, :, 4] = parameters.RenderedVoxelGrid.mu
            average.V[:, :, :4] = -parameters.RenderedVoxelGrid.mu
            average.V[:, :, 4] = parameters.RenderedVoxelGrid.mu

        return average



    def _feature_collapse(self, X):
        """Applied to the feature shoeboxes after extraction"""

        if parameters.VoxletTraining.feature_transform == 'pca':
            return self.model.feature_pca.transform(X.flatten())

        elif parameters.VoxletTraining.feature_transform == 'decimate':
            rate = parameters.VoxletTraining.decimation_rate
            X_sub = X[::rate, ::rate, ::rate]
            return X_sub.flatten()

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

        top_view = np.sum(self.sc.gt_tsdf.V, axis=2)
        plt.imshow(top_view, cmap=plt.get_cmap('Greys'))
        plt.axis('off')

        plt.subplot(133)
        plt.imshow(top_view, cmap=plt.get_cmap('Greys'))

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

        plt.subplot(132).set_xlim(plt.subplot(133).get_xlim())
        plt.subplot(132).set_ylim(plt.subplot(133).get_ylim())

        if savepath:
            plt.savefig(savepath.replace('png', 'pdf'), dpi=400)

    def _get_voxlet_corners(self):

        v_size = parameters.Voxlet.size * np.array(parameters.Voxlet.shape)
        cen = parameters.Voxlet.centre

        c1 = [-cen[0], cen[1]]
        c2 = [cen[0], cen[1]]
        c3 = [cen[0], cen[1] - v_size[1]]
        c4 = [-cen[0], cen[1] - v_size[1]]

        corners = np.array((c2, c1, c4, c3, c2)) / \
            parameters.RenderedVoxelGrid.voxel_size
        return corners[:, ::-1]

    def _plot_voxlet(self, point, normal):
        plt.plot(point[1], point[0], 'or', ms=2)
        scaling = 20.0
        end_point = point + scaling * normal
        plt.plot([point[1], end_point[1]], [point[0], end_point[0]], 'r', lw=1)

        corners = self._get_voxlet_corners()

        norm2 = np.array([-normal[1], normal[0]])
        R = np.vstack((normal, norm2)).T

        t_corners = np.dot(R, corners.T).T + point
        plt.plot(t_corners[:, 1], t_corners[:, 0], '-g', lw=1)


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


# def pool_helper(index, im, vgrid):

#     world_xyz = im.get_world_xyz()
#     world_norms = im.get_world_normals()

#     # convert to linear idx
#     point_idx = index[0] * im.mask.shape[1] + index[1]

#     shoebox = voxel_data.ShoeBox(parameters.Voxlet.shape)  # grid size
#     shoebox.set_p_from_grid_origin(parameters.Voxlet.centre)  # m
#     shoebox.set_voxel_size(parameters.Voxlet.size)  # m
#     shoebox.initialise_from_point_and_normal(
#         world_xyz[point_idx], world_norms[point_idx], np.array([0, 0, 1]))

#     # convert the indices to world xyz space
#     shoebox.fill_from_grid(vgrid)
#     return shoebox.V.flatten()

# # need to import these *after* the pool helper has been defined
# if multiproc:
#     import multiprocessing
#     import functools
#     pool = multiprocessing.Pool(parameters.cores)


# class VoxletExtractor(object):
#     '''
#     extracts voxlets from a voxel grid and a depth image
#     '''
#     def __init__(self):
#         pass

#     def set_voxel_grid(self, vgrid):
#         self.voxel_grid = vgrid

#     def set_image(self, im):
#         self.im = im

#     def extract_voxlets(self, num_to_extract):
#         self
