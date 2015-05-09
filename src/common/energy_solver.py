import numpy as np
from copy import copy
from scipy.ndimage.interpolation import zoom

scale_factor = 0.5

class EnergySolver(object):

    '''
    strange class to solve the final energy function
    '''

    def __init__(self, possible_predictions, accum, alpha):
        # this is a dictionary of all the predictions
        blank_grid = accum.blank_copy()
        blank_grid.countV *= 0
        blank_grid.sumV *= 0

        self.possible_predictions = possible_predictions

        # this is where we will fill all the predictions into
        self.accum = accum
        self.accum.countV *= 0
        self.accum.sumV *= 0
        print "accum has ", self.accum.countV.sum()

        self.alpha = alpha

        # first label each prediction - this ensures we have a unique id for
        # each even if we remove from the dict
        for count, p in enumerate(self.possible_predictions):
            p['voxlet_id'] = count

        # seeing which cells would be full if *all* the cells were full
        print "creating full grid"
        temp = blank_grid.blank_copy()
        for p in self.possible_predictions:
            temp.add_voxlet(p['voxlet'], False, weights=p['weights'])
        self.union_fplan = zoom(temp.countV, scale_factor, order=0) > 0
        self.num_possible_voxels = np.sum(self.union_fplan)
        print "There are %d possible voxels " % self.num_possible_voxels

        # creating now a separate grid for each possible prediction...
        print "creating seperate grids"
        for p in self.possible_predictions:
            temp = blank_grid.blank_copy()
            temp.add_voxlet(p['voxlet'], False, weights=p['weights'])
            p['this_mask_in_world'] = zoom(temp.countV > 0, scale_factor, order=0)

        # here we keep a binary vector which notes which predictions we have used
        self.which_used = np.zeros(len(self.possible_predictions))

        # need to make this in zero, one...
        mu = np.nanmax(self.possible_predictions[0]['voxlet'].V)
        # remember the narrow band distances are alrady RMSE
        self.per_prediction_error = \
            np.array([p['distance'] for p in self.possible_predictions]) / mu

        print "Pred error, min med mean max", \
            np.min(self.per_prediction_error), \
            np.median(self.per_prediction_error), \
            np.mean(self.per_prediction_error), \
            np.max(self.per_prediction_error)


    def solve(self):

        # alpha beta are parameters...
        # hopefully don't complete this loop...

        self.current_prediction_locations = zoom(self.accum.countV, scale_factor, order=0) > 0
        # print "Current cost is %f" % current_cost

        for idx in range(len(self.possible_predictions)):

            current_cost = self.eval_current_union_cost()

            # loop over each possible prediction left and see if want to use it
            costs = np.array([self.eval_potential_union_cost(pred)
                     for pred in self.possible_predictions])

            cost_differences = costs - current_cost

            if np.all(cost_differences >= 0):
                print "Could not add any more! quitting"
                break

            prediction_to_use = np.argmin(cost_differences)
            print "Choosing a prediction with a cost difference of ", cost_differences[prediction_to_use]

            add_this_voxlet = self.possible_predictions.pop(prediction_to_use)
            # print "Adding voxlet"

            # adding to the grid, and also the floorplan
            self.accum.add_voxlet(add_this_voxlet['voxlet'], False, weights=add_this_voxlet['weights'])
            self.current_prediction_locations = zoom(self.accum.countV, scale_factor, order=0) > 0

            # adding to the binary_vector...
            assert(self.which_used[add_this_voxlet['voxlet_id']] == 0)
            self.which_used[add_this_voxlet['voxlet_id']] = 1

        return copy(self.accum.compute_average())

    def eval_current_union_cost(self):

        # should make this in [0, 1]
        total_fit_cost = np.mean(self.which_used * self.per_prediction_error)

        # this is in [0, 1]
        total_filled_cost = \
            1 - np.sum(self.current_prediction_locations).astype(float) / float(self.num_possible_voxels)
        print "costs: %f, %f" % (total_fit_cost, total_filled_cost)

        return self.alpha * total_fit_cost + (1.0 - self.alpha) * total_filled_cost


    def eval_potential_union_cost(self, proposed_addition):
        '''
        evaluates what the union cost would be with the proposed addition
        '''
        temp_used = copy(self.which_used)

        assert(temp_used[proposed_addition['voxlet_id']] == 0)

        temp_used[proposed_addition['voxlet_id']] = 1

        new_fit_cost = np.mean(temp_used * self.per_prediction_error)

        temp_countV = (proposed_addition['this_mask_in_world'] + self.current_prediction_locations)
        new_filled_cost = 1 - np.sum(temp_countV > 0).astype(float) / float(self.num_possible_voxels)

        return self.alpha * new_fit_cost + (1.0 - self.alpha) * new_filled_cost
