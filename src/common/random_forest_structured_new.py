import numpy as np
import time
import cPickle
import pdb
from joblib import Parallel, delayed
from sklearn.decomposition import RandomizedPCA
from scipy.weave import inline
import ipdb


class ForestParams:
    def __init__(self):

        self.num_tests = 10
        self.min_sample_cnt = 2
        self.max_depth = 30
        self.num_trees = 5
        self.bag_size = 0.5
        self.train_parallel = True

        # structured learning params
        #self.pca_dims = 5
        self.num_dims_for_pca = 512  # number of dimensions that pca gets reduced to
        self.sub_sample_exs_pca = True  # can also subsample the number of exs we use for PCA
        self.num_exs_for_pca = 5000

        self.oob_score = False
        self.oob_importance = False

        print 'NOTE: change the params back'

    # def __init__(self):
        # self.num_tests = 2000
        # self.min_sample_cnt = 5
        # self.max_depth = 12
        # self.num_trees = 50
        # self.bag_size = 0.5
        # self.train_parallel = True
        #
        # # structured learning params
        # #self.pca_dims = 5
        # self.num_dims_for_pca = 50 # number of dimensions that pca gets reduced to
        # self.sub_sample_exs_pca = True  # can also subsample the number of exs we use for PCA
        # self.num_exs_for_pca = 5000
        #
        # self.oob_score = False
        # self.oob_importance = False


class Node:

    def __init__(self, node_id, exs_at_node, impurity, probability, medoid_id, tree_id):
        self.node_id = node_id
        #print "In tree %d \t node %d" % (int(tree_id), int(node_id))
        self.exs_at_node = exs_at_node
        self.impurity = impurity
        self.num_exs = float(exs_at_node.shape[0])
        self.is_leaf = True
        self.info_gain = 0.0
        self.tree_id = tree_id

        # just saving the probability of class 1 for now
        self.probability = probability
        self.medoid_id = medoid_id  # if leaf store the medoid id that lands here

    def update_node(self, test_ind, test_thresh, info_gain):
        self.test_ind = test_ind
        self.test_thresh = test_thresh
        self.info_gain = info_gain
        self.is_leaf = False

    def find_medoid_id(self, Y_pca):
        mu = Y_pca.mean(0)
        mu_dist = np.sqrt(((Y_pca - mu[np.newaxis, ...])**2).sum(1))
        return mu_dist.argmin()

    def create_child(self, inds, impurity, prob, y_pca, child_type):

        # TODO make this cleaner, need relative positoin to calculate medoid
        local_inds = np.zeros(inds.shape[0], dtype=np.int64)
        for ii in range(inds.shape[0]):
            local_inds[ii] = np.where(self.exs_at_node == inds[ii])[0][0]
        # find medoid to store at node
        med_id = inds[self.find_medoid_id(y_pca.take(local_inds, 0))]

        if child_type == 'left':
            self.left_node = Node(2*self.node_id+1, inds, impurity, prob, med_id, self.tree_id)
        elif child_type == 'right':
            self.right_node = Node(2*self.node_id+2, inds, impurity, prob, med_id, self.tree_id)

    def get_leaf_nodes(self):
        # returns list of all leaf nodes below this node
        if self.is_leaf:
            return [self]
        else:
            return self.right_node.get_leaf_nodes() + \
                   self.left_node.get_leaf_nodes()

    def test(self, X):
        return X[self.test_ind] < self.test_thresh

    def get_compact_node(self):
        # used for fast forest
        if not self.is_leaf:
            node_array = np.zeros(4)
            # dims 0 and 1 are reserved for indexing children
            node_array[2] = self.test_ind
            node_array[3] = self.test_thresh
        else:
            node_array = np.zeros(2)
            node_array[0] = -1  # indicates that its a leaf
            node_array[1] = self.medoid_id  # the medoid id
        return node_array


class Tree:

    def __init__(self, tree_id, tree_params):
        self.tree_id = tree_id
        self.tree_params = tree_params
        self.num_nodes = 0
        self.label_dims = 0  # dimensionality of label space
        self.compact_tree = None  # used for fast testing forest and small memory footprint

    def build_tree(self, X, Y, node):
        if (node.node_id < ((2**self.tree_params.max_depth)-1)) and (node.impurity > 0.0) \
                and (self.optimize_node(X, Y, node)):
                self.num_nodes += 2
                self.build_tree(X, Y, node.left_node)
                self.build_tree(X, Y, node.right_node)

    def discretize_labels(self, y):

        # perform PCA
        # note this randomly reduces amount of data in Y
        y_pca = self.pca(y)

        # discretize - here binary
        # using PCA based method - alternative is to use kmens
        y_bin = (y_pca[:, 0] > 0).astype('int')

        return y_pca, y_bin

    def pca(self, y):
        # select a random subset of Y dimensions (possibly gives robustness as well as speed)
        rand_dims = np.sort(np.random.choice(y.shape[1], np.minimum(self.tree_params.num_dims_for_pca, y.shape[1]), replace=False))
        y = y.take(rand_dims, 1)
        y_sub = y

        '''
        # optional: select a subset of exs (not so important if PCA is fast)
        if self.tree_params.sub_sample_exs_pca:
            rand_exs = np.sort(np.random.choice(y.shape[0], np.minimum(self.tree_params.num_exs_for_pca, y.shape[0]), replace=False))
            y_sub = y.take(rand_exs, 0)

        # perform PCA
        y_sub_mean = np.mean(y_sub, 0)
        y_sub = y_sub - y_sub_mean
        (l, M) = np.linalg.eig(np.dot(y_sub.T, y_sub))
        y_ds = np.dot(y-y_sub_mean, M[:, 0:self.tree_params.pca_dims])
        '''
        pca = RandomizedPCA(n_components=1) # compute for all components
        y_ds = pca.fit_transform(y_sub)
        return y_ds

    def train(self, X, Y, extracted_from):
        # no bagging
        #exs_at_node = np.arange(Y.shape[0])

        # bagging
        if extracted_from == None:
            exs_at_node = np.random.choice(Y.shape[0], int(Y.shape[0]*self.tree_params.bag_size), replace=False)
        else:
            ids = np.unique(extracted_from)
            ids_for_this_tree = \
                np.random.choice(ids.shape[0], int(ids.shape[0]*self.tree_params.bag_size), replace=False)

            print "ids ", ids.shape
            print "ids_for_this_tree", ids_for_this_tree.shape

            # http://stackoverflow.com/a/15866830/279858
            exs_at_node = []
            for this_id in ids_for_this_tree:
                exs_at_node.append(np.where(extracted_from == this_id)[0])
            exs_at_node = np.unique(np.array(exs_at_node))

            print np.unique(extracted_from[exs_at_node])

            print "exs_at_node ", exs_at_node.shape
            print "exs_at_node ", exs_at_node.dtype

        exs_at_node.sort()
        

        # compute impurity
        #root_prob, root_impurity = self.calc_impurity(0, np.take(Y, exs_at_node), np.ones((exs_at_node.shape[0], 1), dtype='bool'),
        #                                    np.ones(1, dtype='float')*exs_at_node.shape[0])
        # cheating here by putting root impurity to 0.5 - should compute it
        root_prob = 0.5
        root_impurity = 0.5
        root_medoid_id = 0

        # create root
        self.root = Node(0, exs_at_node, root_impurity, root_prob, root_medoid_id, self.tree_id)
        self.num_nodes = 1
        self.label_dims = Y.shape[1]  # dimensionality of label space

        # build tree
        self.build_tree(X, Y, self.root)

        # make compact version for fast testing
        self.compact_tree, _ = self.traverse_tree(self.root, np.zeros(0))

        self.num_feature_dims = X.shape[1]

        if self.tree_params.oob_score:

            # oob score is cooefficient of determintion R^2 of the prediction
            # oob score is in [0, 1], lower values are worse
            # Make predictions for examples not in the bag
            oob_exes = np.setdiff1d(np.arange(Y.shape[0]), exs_at_node)
            pred_idxs = self.test(X[oob_exes, :])

            # Compare the prediction to the GT (must be careful - as only indices returned)
            pred_Y = Y[pred_idxs.astype(np.int32), :]
            gt_Y = Y[oob_exes, :]

            u = ((pred_Y - gt_Y)**2).sum()
            v = ((gt_Y - gt_Y.mean(axis=0))**2).sum()
            self.oob_score = (1- u/v)

            print self.oob_score

    def traverse_tree(self, node, compact_tree_in):
        node_loc = compact_tree_in.shape[0]
        compact_tree = np.hstack((compact_tree_in, node.get_compact_node()))

        # no this assumes that the index for the left and right child nodes are the first two
        if not node.is_leaf:
            compact_tree, compact_tree[node_loc] = self.traverse_tree(node.left_node, compact_tree)
            compact_tree, compact_tree[node_loc+1] = self.traverse_tree(node.right_node, compact_tree)

        return compact_tree, node_loc

    def calc_importance(self):
        ''' borrows from https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/tree/_tree.pyx
        '''

        self.feature_importance = np.zeros(self.num_feature_dims)

        '''visit each node...'''
        stack = []
        node = self.root
        while stack or not node.is_leaf:

            if node.is_leaf:
                node = stack.pop()
            else:
                left = node.left_node
                right = node.right_node

                if not right.is_leaf and not left.is_leaf:
                    self.feature_importance[node.test_ind] += \
                        (node.num_exs * node.impurity -
                        right.num_exs * right.impurity -
                        left.num_exs * left.impurity)

                if not right.is_leaf:
                    stack.append(right)
                node = left

        self.feature_importance /= self.feature_importance.sum()
        return self.feature_importance
            # oob_exes = np.setdiff1d(np.arange(Y.shape[0]), exs_at_node)
            # self.oob_importance = np.zeros((X.shape[1]))

            # # Take each feature dimension in turn
            # for feature_idx in range(X.shape[1]):

            #     # permute this column
            #     to_use = np.random.choice(oob_exes, 100)
            #     this_X = X.copy()[to_use, :]

            #     permutation = np.random.permutation(this_X.shape[0])
            #     this_X[:, feature_idx] = this_X[permutation, feature_idx]

            #     # Compare the prediction to the GT (must be careful - as only indices returned)
            #     pred_Y = Y[self.test(this_X).astype(np.int32), :]
            #     gt_Y = Y[to_use, :]

            #     # maybe here do ssd? Not normalised but could be ok...
            #     self.oob_importance[feature_idx] = \
            #         ((pred_Y - gt_Y)**2).sum(axis=1).mean()

            # print self.oob_importance

    def test(self, X):
        op = np.zeros(X.shape[0])
        # check out apply() in tree.pyx in scikitlearn

        # single dim test
        for ex_id in range(X.shape[0]):
            node = self.root
            while not node.is_leaf:
                if X[ex_id, node.test_ind] < node.test_thresh:
                    node = node.right_node
                else:
                    node = node.left_node
            # return medoid id
            op[ex_id] = node.medoid_id
        return op

    def test_fast(self, X):
        op = np.zeros((X.shape[0]))
        tree = self.compact_tree  # work around as I don't think I can pass self.compact_tree

        #in memory: for non leaf  node - 0 is lchild index, 1 is rchild, 2 is dim to test, 3 is threshold
        #in memory: for leaf node - 0 is leaf indicator -1, 1 is the medoid id
        code = """
        int ex_id, node_loc;
        for (ex_id=0; ex_id<NX[0]; ex_id++) {
            node_loc = 0;
            while (tree[node_loc] != -1) {
                if (X2(ex_id, int(tree[node_loc+2]))  <  tree[node_loc+3]) {
                    node_loc = tree[node_loc+1];  // right node
                }
                else {
                    node_loc = tree[node_loc];  // left node
                }

            }

            op[ex_id] = tree[node_loc + 1];  // medoid id

        }
        """
        inline(code, ['X', 'op', 'tree'])
        return op

    def leaf_nodes(self):
        '''returns list of all leaf nodes'''
        return self.root.get_leaf_nodes()


    def generate_candidate_splits(self, X, exs_at_node):
        """
        Generates random indices and thresholds for testing.
        """
        # TODO speed this up

        # indices
        test_inds = np.sort(np.random.random_integers(0, X.shape[1]-1, self.tree_params.num_tests))

        # compute min and max
        x_min = np.zeros(self.tree_params.num_tests)
        x_max = np.zeros(self.tree_params.num_tests)
        for tt in range(self.tree_params.num_tests):
            x_col = X[exs_at_node, test_inds[tt]]
            x_min[tt] = x_col.min(0)
            x_max[tt] = x_col.max(0)

        # threshold
        test_thresh = (x_max - x_min)*np.random.random_sample(self.tree_params.num_tests) + x_min

        return test_inds, test_thresh

    def compute_splits(self, X, y_bin_local, exs_at_node, test_inds, test_thresh):
        left_count = np.zeros((2, test_inds.shape[0]))
        right_count = np.zeros((2, test_inds.shape[0]))

        code = """
        int num_tests = Ntest_inds[0];
        int num_exs = Nexs_at_node[0];
        int ex_id, test_id;

        for (test_id=0; test_id<num_tests; test_id++) {

            for (ex_id=0; ex_id<num_exs; ex_id++) {
                if (X2(exs_at_node[ex_id], test_inds[test_id]) < test_thresh[test_id]) {
                    // right
                    RIGHT_COUNT2(y_bin_local[ex_id], test_id) += 1;  // y is current rel, not abs indexing
                }
                else {
                    // left
                    LEFT_COUNT2(y_bin_local[ex_id], test_id) += 1;  // y is current rel, not abs indexing
                }

            }

        }
        """
        inline(code, ['X', 'y_bin_local', 'exs_at_node', 'test_inds', 'test_thresh', 'left_count', 'right_count'])

        return left_count, right_count

    def get_indices_of_split(self, X, exs_at_node, test_ind, test_thresh):
        left_inds = np.ones(exs_at_node.shape[0], dtype=int)*-1
        right_inds = np.ones(exs_at_node.shape[0], dtype=int)*-1

        code = """
        int num_exs = Nexs_at_node[0];
        int ex_id;

        for (ex_id=0; ex_id<num_exs; ex_id++) {
            if (X2(exs_at_node[ex_id], test_ind) < test_thresh) {
                // right
                right_inds[ex_id] = exs_at_node[ex_id];
            }
            else {
                // left
                left_inds[ex_id] = exs_at_node[ex_id]; // TODO replace this back
            }

        }
        """
        inline(code, ['X', 'exs_at_node', 'test_ind', 'test_thresh', 'left_inds', 'right_inds'])

        # discard the empty dimensions and sort (sort might be unnecessary)
        left_inds = left_inds[np.where(left_inds != -1)[0]]
        right_inds = right_inds[np.where(right_inds != -1)[0]]
        left_inds.sort()
        right_inds.sort()

        return left_inds, right_inds

    def calc_impurity(self, node_cnt):
        # TODO currently num_exs is changed to deal with divide by zero, fix this

        num_exs = node_cnt.sum(0)

        # if don't want to divide by 0 so add a 1 to the numerator
        invalid_inds = np.where(num_exs == 0.0)[0]
        num_exs[invalid_inds] = 1

        # estimate probability
        # just binary classification
        prob = node_cnt / num_exs
        prob[:, invalid_inds] = 0.5  # 1/num_classes

        # binary classification
        #impurity = -np.sum(prob*np.log2(prob))  # entropy
        impurity = 1-(prob*prob).sum(0)  # gini

        num_exs[invalid_inds] = 0.0
        return prob, impurity, num_exs

    def optimize_node(self, X, Y, node):

        # discretize label space
        # TODO fix this: dont want to make copy of Y
        y_local = np.take(Y, node.exs_at_node, 0)
        y_pca, y_bin = self.discretize_labels(y_local)

        # generate potential tests i.e. splits
        test_inds, test_thresh = self.generate_candidate_splits(X, node.exs_at_node)

        # for each random test, count the exs that go left and right
        left_count, right_count = self.compute_splits(X, y_bin, node.exs_at_node, test_inds, test_thresh)

        # compute impurity
        prob_l, impurity_l, num_exs_l = self.calc_impurity(left_count)
        prob_r, impurity_r, num_exs_r = self.calc_impurity(right_count)
        valid_inds = (num_exs_l >= self.tree_params.min_sample_cnt) & (num_exs_r >= self.tree_params.min_sample_cnt)

        successful_split = False
        if valid_inds.sum() > 0:

            # information gain - want the minimum
            num_exs_l_norm = num_exs_l/node.num_exs
            num_exs_r_norm = num_exs_r/node.num_exs
            #info_gain = - node.impurity + (num_exs_r_norm*impurity_r) + (num_exs_l_norm*impurity_l)
            info_gain = (num_exs_r_norm*impurity_r) + (num_exs_l_norm*impurity_l)

            # make sure we con only select from valid splits
            info_gain[~valid_inds] = info_gain.max() + 10e-10  # plus small constant
            best_split = info_gain.argmin()

            # if the info gain is acceptable split the node
            # TODO is this the best way of checking info gain?
            #if info_gain[best_split] > self.tree_params.min_info_gain:
            # create new child nodes and update current node

            # get the ids of the nodes going left and right for the best split
            left_inds, right_inds = self.get_indices_of_split(X, node.exs_at_node, int(test_inds[best_split]), float(test_thresh[best_split]))

            node.update_node(test_inds[best_split], test_thresh[best_split], info_gain[best_split])
            node.create_child(left_inds, impurity_l[best_split], prob_l[1, best_split], y_pca, 'left')
            node.create_child(right_inds, impurity_r[best_split], prob_r[1, best_split], y_pca, 'right')

            successful_split = True

        return successful_split


## Parallel training helper - used to train trees in parallel
def train_forest_helper(t_id, X, Y, extracted_from, params, seed):
    #print 'tree', t_id
    np.random.seed(seed)
    tree = Tree(t_id, params)
    tree.train(X, Y, extracted_from)
    return tree


class Forest:

    def __init__(self, params):
        self.params = params
        self.trees = []

    #def save(self, filename):
        # TODO make lightweight version for saving
        #with open(filename, 'wb') as fid:
        #    cPickle.dump(self, fid)

    def train(self, X, Y, extracted_from=None):
        '''
        extracted_from is an optional array which defines a class label
        for each training example. if provided, the bagging is done at
        the level of np.unique(extracted_from)
        '''
        if np.any(np.isnan(X)):
            raise Exception('nans should not be present in training X')

        if np.any(np.isnan(Y)):
            raise Exception('nans should not be present in training Y')

        if self.params.train_parallel:
            # TODO Can I make this faster by sharing the data?
            #print 'Parallel training'
            # need to seed the random number generator for each process
            seeds = np.random.random_integers(0, np.iinfo(np.int32).max, self.params.num_trees)
            self.trees.extend(Parallel(n_jobs=-1)
                (delayed(train_forest_helper)(t_id, X, Y, extracted_from, self.params, seeds[t_id])
                                             for t_id in range(self.params.num_trees)))
        else:
            #print 'Standard training'
            for t_id in range(self.params.num_trees):
                #print 'tree', t_id
                tree = Tree(t_id, self.params)
                tree.train(X, Y, extracted_from)
                self.trees.append(tree)
        #print 'num trees ', len(self.trees)

    def test(self, X):
        if np.any(np.isnan(X)):
            raise Exception('nans should not be present in test X')

        # return the medoid id at each leaf
        op = np.zeros((X.shape[0], len(self.trees)))
        for tt, tree in enumerate(self.trees):
            #op[:, tt] = tree.test(X)  # old slow way
             op[:, tt] = tree.test_fast(X)
        return op

    def delete_trees(self):
        del self.trees[:]

    def calc_importance(self):
        imp = [tree.calc_importance() for tree in self.trees]
        return np.vstack(imp).mean(axis=0)
