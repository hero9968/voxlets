import numpy as np
import sys
sys.path.append('../../common')
import voxel_data
import paths
from scipy.stats import mode

class Reconstructer(object):
    '''
    does the final prediction
    '''

    def __init__(self, reconstruction_type, combine_type):
        self.reconstruction_type = reconstruction_type
        self.combine_type = combine_type


    def set_km_dict(self, km):
        self.km = km


    def set_pca_comp(self, pca):
        self.pca = pca


    def set_forest(self, forest):
        self.forest = forest


    def set_test_im(self, test_im):
        self.im = test_im


    def sample_points(self, num_to_sample):
        '''
        sampling points from the test image
        '''
        test_mask = ~np.isnan(self.im.frontrender)

        indices = np.array(np.nonzero(test_mask)).T
        np.random.seed(1)
        samples = np.random.randint(0, indices.shape[0], num_to_sample)
        self.sampled_idxs = indices[samples, :]
        #print "Sampled these many points : " + str(self.sampled_idxs.shape)


    def classify_features(self, features):
        '''
        classifying the features
        here will be where the pca etc is used if it is used...
        '''
        pred_classes = self.forest.predict(features)
        #probs = self.forest.predict_proba(features)
        #small_number = 0.0001
        #entropy = -np.sum(probs+small_number * np.log(probs+small_number), axis=1)
        
        temp_votes = np.array([tree.predict(features) for tree in self.forest.estimators_]).T
        per_tree_class_predictions = self.forest.classes_[temp_votes.astype(int)]

        return (pred_classes, per_tree_class_predictions)


    def _initialise_voxlet(self, index):
        '''
        given a point in an image, creates a new voxlet at an appropriate
        position and rotation in world space
        '''

        assert(index.shape[0]==2)

        # getting the xyz and normals in world space
        world_xyz = self.im.get_world_xyz()
        world_norms = self.im.get_world_normals()

        # convert to linear idx
        point_idx = index[0] * self.im.mask.shape[1] + index[1]

        # creating the voxlet
        shoebox = voxel_data.ShoeBox(paths.voxlet_shape) # grid size
        shoebox.set_p_from_grid_origin(paths.voxlet_centre) #m
        shoebox.set_voxel_size(paths.voxlet_size) #m
        shoebox.initialise_from_point_and_normal(world_xyz[point_idx], 
                                                 world_norms[point_idx], 
                                                 np.array([0, 0, 1]))
        return shoebox


    def _convert_forest_prediction_to_voxlet_vector(self, forest_prediction):

        if self.reconstruction_type == 'standard_kmeans':

            return self.km.cluster_centers_[forest_prediction]

        elif self.reconstruction_type == 'kmeans_on_pca':

            # look up the basis vectors for this point
            basis_vectors = self.km.cluster_centers_[forest_prediction]
            return self.pca.inverse_transform(basis_vectors)


    def _reconstruct_voxlet_V(self, pred_class, per_tree_class_predictions):
        '''
        given the forest output reconstructs the appropriate voxlet 
        '''

        if self.combine_type == 'modal_vote':

            voxlet_as_vector = self._convert_forest_prediction_to_voxlet_vector(pred_class)


        elif self.combine_type == 'medioid' and self.reconstruction_type == 'kmeans_on_pca':
            
            # getting all the basis vectors predicted from each tree in the forest
            voxlets_as_basis_vectors = np.array([self.km.cluster_centers_[pred_class]
                                            for pred_class in per_tree_class_predictions])

            # finding the medioid prediction in pca space
            mu = voxlets_as_basis_vectors.mean(0)
            mu_dist = np.sqrt(((voxlets_as_basis_vectors - mu[np.newaxis, ...])**2).sum(1))
            median_item_idx = mu_dist.argmin()

            medioid_basis_vector = voxlets_as_basis_vectors[median_item_idx]

            # convert this basis vector to full representation
            voxlet_as_vector = self.pca.inverse_transform(medioid_basis_vector)

            #voxlet_as_vector = self.pca.inverse_transform(np.mean(voxlets_as_basis_vectors, axis=0))

        else:
            error("Dont know this method")

        return voxlet_as_vector.reshape(paths.voxlet_shape)


    def initialise_output_grid(self, method='from_image', gt_grid=None):


        if method == 'from_image':
            '''initialising grid purely based on image data'''

            # just get the points we care about
            world_xyz = self.im.get_world_xyz()[self.im.mask.flatten()==1, :]

            # setting grid size
            grid_origin = np.percentile(world_xyz, 5, axis=0) - 0.15
            grid_end = np.percentile(world_xyz, 95, axis=0) + 0.15

        elif method == 'from_grid':
            '''initialising based on the ground truth grid'''
            assert(gt_grid!=None)

            # pad the gt grid slightly
            grid_origin = gt_grid.origin - 0.05
            grid_end = gt_grid.origin + np.array(gt_grid.V.shape).astype(float) * gt_grid.vox_size + 0.05

        voxlet_size = paths.voxlet_size/2.0
        grid_dims_in_real_world = grid_end - grid_origin
        V_shape = (grid_dims_in_real_world / (voxlet_size)).astype(int)

        self.accum = voxel_data.UprightAccumulator(V_shape)
        self.accum.set_origin(grid_origin)
        self.accum.set_voxel_size(voxlet_size)


    def fill_in_output_grid(self, max_points=500, reconstruction_type='kmeans_on_pca'):
        '''
        doing the final reconstruction
        vgrid is th e ground truth grid for size and shape - will have to change this soon!
        '''

        "extract features from test image"
        combined_features = self.im.get_features(self.sampled_idxs)

        "classify according to the forest"
        forest_predictions, per_tree_class_predictions = \
            self.classify_features(combined_features)

        "creating the output voxel grid"
        if self.accum == None:
            error('have not initilaised the accumulator')

        "for each forest prediction, do something sensible"
        order_to_fill = range(max_points)

        # loop over all the predictions
        for count, idx_idx in enumerate(order_to_fill):

            # create the shoebox for this point
            shoebox = self._initialise_voxlet(self.sampled_idxs[idx_idx])

            # look up the prediction result
            forest_prediction = forest_predictions[idx_idx]
            per_tree_class_prediction = per_tree_class_predictions[idx_idx]

            # adding the shoebox into the result
            shoebox.V = self._reconstruct_voxlet_V(forest_prediction, per_tree_class_prediction)
            self.accum.add_voxlet(shoebox)

            if np.mod(count, 100) == 0:
                print "Added shoebox " + str(count)

            if count > max_points:
                print "Ending"
                break

        return self.accum


    def fill_in_output_grid_oma(self, max_points=500, special=None):
        '''
        doing the final reconstruction
        vgrid is th e ground truth grid for size and shape - will have to change this soon!

        OMA forest saved is actually a dict conprising of the forest and the index data
        '''

        "extract features from test image"
        combined_features = self.im.get_features(self.sampled_idxs)

        "Special instruction to only compute some features"
        if special=='cobweb':
            combined_features = combined_features[:, :32]

        "classify according to the forest"
        forest_predictions = self.forest['forest'].test(combined_features)
        print "Forest predictons has shape " + str(forest_predictions.shape)

        "creating the output voxel grid"
        if self.accum == None:
            error('have not initilaised the accumulator')

        "for each forest prediction, do something sensible"
        order_to_fill = range(max_points)

        # loop over all the predictions
        for count, idx_idx in enumerate(order_to_fill):

            # create the shoebox for this point
            shoebox = self._initialise_voxlet(self.sampled_idxs[idx_idx])

            # look up the structured prediction from the forest - this gives the indices into the original data
            forest_prediction = forest_predictions[idx_idx]

            # for now just use the modal prediction
            this_class = int(mode(forest_prediction)[0])
            this_pca_representation = self.forest['traindata'][this_class, :]

            # convert this to a full voxlet
            full_V = self.forest['pca_model'].inverse_transform(this_pca_representation)

            # adding the shoebox into the result
            shoebox.V = full_V.reshape(paths.voxlet_shape)
            self.accum.add_voxlet(shoebox)

            if np.mod(count, 100) == 0:
                print "Added shoebox " + str(count)

            if count > max_points:
                print "Ending"
                break

        return self.accum

