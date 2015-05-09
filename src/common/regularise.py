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


