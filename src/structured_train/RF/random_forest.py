import numpy as np
import time
import cPickle
import pdb
from joblib import Parallel, delayed

class ForestParams:
    def __init__(self):
        self.num_tests = 200
        self.min_sample_cnt = 2
        #self.max_depth = 30.0
        self.max_depth = 10
        self.num_trees = 20
        self.bag_size = 0.1
        self.train_parallel = False


class Node:

    def __init__(self, node_id, exs_at_node, impurity, probability):
        self.node_id = node_id
        self.exs_at_node = exs_at_node
        self.impurity = impurity
        self.num_exs = float(exs_at_node.shape[0])
        self.is_leaf = True
        self.info_gain = 0.0

        # just saving the probability of class 1 for now
        self.probability = probability

    def update_node(self, test_ind1, test_thresh, info_gain):
        self.test_ind1 = test_ind1
        self.test_thresh = test_thresh
        self.info_gain = info_gain

    def create_children(self, test_res, impurity_l, prob_l, impurity_r, prob_r):
        left_inds = self.exs_at_node[np.where(~test_res)[0]]
        right_inds = self.exs_at_node[np.where(test_res)[0]]

        self.left_node = Node(2*self.node_id+1, left_inds, impurity_l, prob_l)
        self.right_node = Node(2*self.node_id+2, right_inds, impurity_r, prob_r)
        self.is_leaf = False

    def test(self, X):
        return X[self.test_ind1] < self.test_thresh


class Tree:

    def __init__(self, tree_id, tree_params):
        self.tree_id = tree_id
        self.tree_params = tree_params
        self.num_nodes = 0

    def build_tree(self, X, Y, node):
        if (node.node_id < ((2**self.tree_params.max_depth)-1))  \
                and (self.optimize_node(np.take(X, node.exs_at_node, 0), np.take(Y, node.exs_at_node), node)):
                self.num_nodes += 2
                self.build_tree(X, Y, node.left_node)
                self.build_tree(X, Y, node.right_node)

    def train(self, X, Y):
        # no bagging
        #exs_at_node = np.arange(Y.shape[0])

        # bagging
        exs_at_node = np.random.choice(Y.shape[0], int(Y.shape[0]*self.tree_params.bag_size), replace=False)
        exs_at_node.sort()

        # compute impurity
        prob, impurity = self.calc_impurity(0, np.take(Y, exs_at_node), np.ones((exs_at_node.shape[0], 1), dtype='bool'),
                                            np.ones(1, dtype='float')*exs_at_node.shape[0])

        # create root
        #print np.take(Y, exs_at_node).shape
        #print prob.shape
        self.root = Node(0, exs_at_node, impurity, float(prob[0]))
        self.num_nodes = 1

        # build tree
        self.build_tree(X, Y, self.root)

    def test(self, X):
        op = np.zeros((X.shape[0]))
        # check out apply() in tree.pyx in scikitlearn

        # single dim test
        for ex_id in range(X.shape[0]):
            node = self.root
            while not node.is_leaf:
                if X[ex_id, node.test_ind1] < node.test_thresh:
                    node = node.right_node
                else:
                    node = node.left_node
            op[ex_id] = node.probability
        return op

    def calc_impurity(self, node_id, y_local, test_res, num_exs):
        # TODO currently num_exs is changed to deal with divide by zero, fix this
        # if don't want to divide by 0 so add a 1 to the numerator

        number_tests = test_res.shape[1]

        # MF - invalid if all data on one side
        invalid_inds = np.where(num_exs == 0.0)[0]
        num_exs[invalid_inds] = 1

        # means (serious use of broadcasting!)
        node_test = test_res * (y_local[:, np.newaxis])
        Y_col_sums = np.sum(node_test, axis=0)
        means = Y_col_sums / num_exs

        # variance
        deviation_from_mean = test_res * (y_local[:, np.newaxis] - means) # employs broadcasting
        variances = np.sum(deviation_from_mean**2, axis=0) / num_exs
        
        counts = np.sum(test_res, axis=0)
        
        return means, variances

    def node_split(self, x_local):
        # left node is false, right is true
        # single dim test

        # MF - choosing which... data to test on?
        # number of rows = num data points, num cols = number tests
        # x_local.shape[1]-1 --> dimension of data
        test_inds_1 = np.sort(np.random.random_integers(0, x_local.shape[1]-1, self.tree_params.num_tests))
        x_local_expand = x_local.take(test_inds_1, 1)

        # MF - choosing a set of possible test thresholds, in [x_min, x_max]
        x_min = x_local_expand.min(0)
        x_max = x_local_expand.max(0)
        test_thresh = (x_max - x_min)*np.random.random_sample(self.tree_params.num_tests) + x_min
        #valid_var = (x_max != x_min)

        test_res = x_local_expand < test_thresh

        return test_res, test_inds_1, test_thresh

        # MF - on larger data could take random subset, or 

    def optimize_node(self, x_local, y_local, node):
        # TODO if num_tests is very large could loop over test_res in batches
        # TODO is the number of invalid splits is small it might be worth deleting the corresponding tests
        # %timeit rf.trees[0].optimize_node(X, Y, rf.trees[0].root)
        # MF note - if unsuccessful split, this branch will be stopped! Watch out...

        # perform split at node
        # MF - returns results (binary), features, thresholds
        test_res, test_inds1, test_thresh = self.node_split(x_local)

        # count examples left and right
        num_exs_l = (~test_res).sum(axis=0).astype('float')
        num_exs_r = x_local.shape[0] - num_exs_l  # i.e. num_exs_r = test_res.sum(axis=0).astype('float')
        valid_inds = (num_exs_l >= self.tree_params.min_sample_cnt) & (num_exs_r >= self.tree_params.min_sample_cnt)

        successful_split = False
        if valid_inds.sum() > 0:
            # child node impurity
            # MF - for each different split!
            prob_l, impurity_l = self.calc_impurity(node.node_id, y_local, ~test_res, num_exs_l)
            prob_r, impurity_r = self.calc_impurity(node.node_id, y_local, test_res, num_exs_r)

             # information gain - want the minimum
            num_exs_l_norm = num_exs_l/node.num_exs
            num_exs_r_norm = num_exs_r/node.num_exs
            #info_gain = - node.impurity + (num_exs_r_norm*impurity_r) + (num_exs_l_norm*impurity_l)
            #temp, initial_imp = self.calc_impurity(node.node_id, y_local, 0*test_res+1, num_exs_r+num_exs_l)
            info_gain = -((num_exs_r_norm*impurity_r) + (num_exs_l_norm*impurity_l))
            
            # make sure we con only select from valid splits
            info_gain[~valid_inds] = info_gain.min() - 10e-10  # plus small constant
            #print info_gain
            best_split = info_gain.argmax() # MF - index of test with best result

            # if the info gain is acceptable split the node
            # TODO is this the best way of checking info gain?
            #if info_gain[best_split] > self.tree_params.min_info_gain:
            # create new child nodes and update current node
           

            node.update_node(test_inds1[best_split], test_thresh[best_split], info_gain[best_split])
            node.create_children(test_res[:, best_split], impurity_l[best_split], prob_l[best_split],
                                 impurity_r[best_split], prob_r[best_split])
            successful_split = True

        return successful_split


## Parallel training helper - used to train trees in parallel
def train_forest_helper(t_id, X, Y, params, seed):
    #print 'tree', t_id
    np.random.seed(seed)
    tree = Tree(t_id, params)
    tree.train(X, Y)
    return tree


class Forest:

    def __init__(self, params):
        self.params = params
        self.trees = []

    #def save(self, filename):
        # TODO make lightweight version for saving
        #with open(filename, 'wb') as fid:
        #    cPickle.dump(self, fid)

    def train(self, X, Y):
        if self.params.train_parallel:
            # TODO Can I make this faster by sharing the data?
            #print 'Parallel training'
            # need to seed the random number generator for each process
            seeds = np.random.random_integers(0, np.iinfo(np.int32).max, self.params.num_trees)
            self.trees.extend(Parallel(n_jobs=-1)(delayed(train_forest_helper)(t_id, X, Y, self.params, seeds[t_id])
                                             for t_id in range(self.params.num_trees)))
        else:
            #print 'Standard training'
            for t_id in range(self.params.num_trees):
                #print 'tree', t_id
                tree = Tree(t_id, self.params)
                tree.train(X, Y)
                self.trees.append(tree)
        #print 'num trees ', len(self.trees)

    def test(self, X):
        op = np.zeros(X.shape[0])
        for tt, tree in enumerate(self.trees):
            op_local = tree.test(X)
            op += op_local
        op /= float(len(self.trees))
        return op

    def delete_trees(self):
        del self.trees[:]
