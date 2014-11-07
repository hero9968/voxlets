import numpy as np
import sys
sys.path.append('../../common')
import voxel_data
import paths

class Reconstructer(object):
    '''
    does the final prediction
    '''

    def __init__(self):
        pass


    def set_km_dict(self, km):
        self.km = km


    def set_pca_comp(self, pca):
        self.pca = pca


    def set_forest(self, forest):
        self.forest = forest


    def set_test_im(self, test_im):
        self.im = test_im


    def sample_points(self, num_to_sample):
        ''' sampling points from the test image
        '''
        test_mask = ~np.isnan(self.im.frontrender)

        indices = np.array(np.nonzero(test_mask)).T
        np.random.seed(1)
        samples = np.random.randint(0, indices.shape[0], num_to_sample)
        self.sampled_idxs = indices[samples, :]
        print "Sampled these many points : " + str(self.sampled_idxs.shape)


    def classify_features(self, features):
        '''
        classifying the features
        here will be where the pca etc is used if it is used...
        '''
        pred_classes = self.forest.predict(features)
        probs = self.forest.predict_proba(features)
        small_number = 0.0001
        entropy = -np.sum(probs+small_number * np.log(probs+small_number), axis=1)
        print "Max entropy is " + str(np.max(entropy))

        print "Pred classes has shape " + str(pred_classes.shape)
        print "Probs has shape " + str(probs.shape)
        print "Entropy has shape " + str(entropy.shape)

        #modal_classes = mode(tree_predictions, axis=1)[0]
        #print np.array(modal_classes).shape
        #agree_with_modal = modal_classes == tree_predictions
        #confidences = np.sum(agree_with_modal, axis=1).astype(float) / float(len(forest.estimators_))
        #print "confidences has shape " + str(confidences.shape)

        return (pred_classes, probs, entropy)


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


    def _reconstruct_voxlet_V(self, pred_class, method):
        '''
        given the forest output reconstructs the appropriate voxlet 
        '''

        if reconstruction_type == 'standard_kmeans':

            voxlet_as_vector = self.km.cluster_centers_[pred_class]

        elif reconstruction_type == 'kmeans_on_pca':

            # look up the basis vectors for this point
            basis_vectors = self.km.cluster_centers_[pred_class]
            voxlet_as_vector = self.pca.inverse_transform(basis_vectors)

        return voxlet_as_vector.reshape(paths.voxlet_shape)

        # forming a medioid of all the voxlets in this cluster...    
        #all_indices = tree_predictions[idx_idx]
        #cluster_center = km.cluster_centers_[int(all_indices[0])]


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


    def fill_in_output_grid(self, max_points=20, reconstruction_type='kmeans_on_pca'):
        '''
        doing the final reconstruction
        vgrid is th e ground truth grid for size and shape - will have to change this soon!
        '''

        "extract features from test image"
        combined_features = self.im.get_features(self.sampled_idxs)
        print "Combined f has shape " + str(combined_features.shape)

        "classify according to the forest"
        forest_predictions, class_probs, entropy = \
            self.classify_features(combined_features)
        
        "creating the output voxel grid"
        if self.accum == None:
            error('have not initilaised the accumulator')

        "for each forest prediction, do something sensible"

        # fill in reverse order of confidence
        order_to_fill = entropy.argsort()[::-1]

        # loop over all the predictions
        for count, idx_idx in enumerate(order_to_fill):

            # create the shoebox for this point
            shoebox = self._initialise_voxlet(self.sampled_idxs[idx_idx])

            # look up the prediction result
            forest_prediction = forest_predictions[idx_idx]

            shoebox.V = self._reconstruct_voxlet_V(forest_prediction, reconstruction_type)
                        
            self.accum.add_voxlet(shoebox)

            if np.mod(count, 10) == 0:
                print "Added shoebox " + str(count)

            if count > max_points:
                print "Ending"
                break

        return self.accum


