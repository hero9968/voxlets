import numpy as np
import time
import cPickle
import pdb
from joblib import Parallel, delayed
from sklearn.decomposition import RandomizedPCA


class ForestParams:
    def __init__(self):
        self.num_tests = 500
        self.min_sample_cnt = 5
        self.max_depth = 25
        self.num_trees = 40
        self.bag_size = 0.5
        self.train_parallel = False
        self.njobs = 8

        # structured learning params
        #self.pca_dims = 5
        self.num_dims_for_pca = 100 # number of dimensions that pca gets reduced to
        self.sub_sample_exs_pca = True  # can also subsample the number of exs we use for PCA
        self.num_exs_for_pca = 2500

        self.oob_score = True
        self.oob_importance = False


class Node:

    def __init__(self, node_id, exs_at_node, impurity, probability, medoid_id, tree_id):
        self.node_id = node_id
        depth = np.floor(np.log2(node_id+1))
        # print "In tree %d \t node %d \t depth %d" % (int(tree_id), int(node_id), int(depth))
        self.exs_at_node = exs_at_node
        self.impurity = impurity
        self.num_exs = float(exs_at_node.shape[0])
        self.is_leaf = True
        self.info_gain = 0.0
        self.tree_id = tree_id

        # just saving the probability of class 1 for now
        self.probability = probability
        self.medoid_id = medoid_id

    def update_node(self, test_ind1, test_thresh, info_gain):
        self.test_ind1 = test_ind1
        self.test_thresh = test_thresh
        self.info_gain = info_gain
        self.is_leaf = False

    def find_medoid_id(self, y_local):
        mu = y_local.mean(0)
        mu_dist = np.sqrt(((y_local - mu[np.newaxis, ...])**2).sum(1))
        return mu_dist.argmin()

    def create_child(self, test_res, impurity, prob, y_local, child_type):
        # test_res is binary and the same shape[0] as y_local
        assert test_res.shape[0] == y_local.shape[0]
        assert self.exs_at_node.shape[0] == y_local.shape[0]

        # save absolute location in dataset
        inds_local = np.where(test_res)[0]
        inds = self.exs_at_node[inds_local]

        # work out which values of y will be at the child node, then take the medoid
        med_id = inds[self.find_medoid_id(y_local.take(inds_local, 0))]

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
        return X[self.test_ind1] < self.test_thresh


class Tree:

    def __init__(self, tree_id, tree_params):
        self.tree_id = tree_id
        self.tree_params = tree_params
        self.num_nodes = 0
        self.label_dims = 0  # dimensionality of label space

    def build_tree(self, X, Y, node):
        if (node.node_id < ((2**self.tree_params.max_depth)-1)) and (node.impurity > 0.0) \
                and (self.optimize_node(np.take(X, node.exs_at_node, 0), np.take(Y, node.exs_at_node, 0), node)):
                self.num_nodes += 2
                self.build_tree(X, Y, node.left_node)
                self.build_tree(X, Y, node.right_node)
        else:
            depth = np.floor(np.log2(node.node_id+1))
            # print "Leaf node: In tree %d \t depth %d \t %d examples" % \
            #     (int(self.tree_id), int(depth), node.exs_at_node.shape[0])

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
        num_to_sample = int(float(Y.shape[0])*self.tree_params.bag_size)

        if extracted_from is None:
            exs_at_node = np.random.choice(Y.shape[0], num_to_sample, replace=False)
        else:
            ids = np.unique(extracted_from)
            ids_for_this_tree = \
                np.random.choice(ids.shape[0], int(float(ids.shape[0])*self.tree_params.bag_size), replace=False)

            # http://stackoverflow.com/a/15866830/279858
            exs_at_node = []
            for this_id in ids_for_this_tree:
                exs_at_node.append(np.where(extracted_from == this_id)[0])
            exs_at_node = np.hstack(exs_at_node)

            exs_at_node = np.unique(np.array(exs_at_node))

            if exs_at_node.shape[0] > num_to_sample:
                exs_at_node = np.random.choice(exs_at_node, num_to_sample, replace=False)

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
                    self.feature_importance[node.test_ind1] += \
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



    def test(self, X, max_depth=np.inf):
        op = np.zeros(X.shape[0])
        # check out apply() in tree.pyx in scikitlearn

        # single dim test
        for ex_id in range(X.shape[0]):
            node = self.root
            depth = 0
            while not (node.is_leaf or depth >= max_depth):
                if X[ex_id, node.test_ind1] < node.test_thresh:
                    node = node.right_node
                else:
                    node = node.left_node
                depth += 1
            # return medoid id
            op[ex_id] = node.medoid_id
        return op

    def leaf_nodes(self):
        '''returns list of all leaf nodes'''
        return self.root.get_leaf_nodes()

    def calc_impurity(self, node_id, y_bin, test_res, num_exs):
        # TODO currently num_exs is changed to deal with divide by zero, fix this
        # if don't want to divide by 0 so add a 1 to the numerator
        invalid_inds = np.where(num_exs == 0.0)[0]
        num_exs[invalid_inds] = 1

        # estimate probability
        # just binary classification
        node_test = test_res * (y_bin[:, np.newaxis] == 1)

        prob = np.zeros((2, test_res.shape[1]))
        prob[1, :] = node_test.sum(axis=0) / num_exs
        prob[0, :] = 1 - prob[1, :]
        prob[:, invalid_inds] = 0.5  # 1/num_classes

        # binary classification
        #impurity = -np.sum(prob*np.log2(prob))  # entropy
        impurity = 1-(prob*prob).sum(0)  # gini

        num_exs[invalid_inds] = 0.0
        return prob, impurity

    def node_split(self, x_local):
        # left node is false, right is true
        # single dim test
        test_inds_1 = np.sort(np.random.random_integers(0, x_local.shape[1]-1, self.tree_params.num_tests))
        x_local_expand = x_local.take(test_inds_1, 1)
        x_min = x_local_expand.min(0)
        x_max = x_local_expand.max(0)
        test_thresh = (x_max - x_min)*np.random.random_sample(self.tree_params.num_tests) + x_min
        #valid_var = (x_max != x_min)

        test_res = x_local_expand < test_thresh

        return test_res, test_inds_1, test_thresh

    def optimize_node(self, x_local, y_local, node):
        # TODO if num_tests is very large could loop over test_res in batches
        # TODO is the number of invalid splits is small it might be worth deleting the corresponding tests

        # perform split at node
        test_res, test_inds1, test_thresh = self.node_split(x_local)

        # discretize label space
        y_pca, y_bin = self.discretize_labels(y_local)

        # count examples left and right
        num_exs_l = (~test_res).sum(axis=0).astype('float')
        num_exs_r = x_local.shape[0] - num_exs_l  # i.e. num_exs_r = test_res.sum(axis=0).astype('float')
        valid_inds = (num_exs_l >= self.tree_params.min_sample_cnt) & (num_exs_r >= self.tree_params.min_sample_cnt)

        successful_split = False
        if valid_inds.sum() > 0:
            # child node impurity
            prob_l, impurity_l = self.calc_impurity(node.node_id, y_bin, ~test_res, num_exs_l)
            prob_r, impurity_r = self.calc_impurity(node.node_id, y_bin, test_res, num_exs_r)

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
            node.update_node(test_inds1[best_split], test_thresh[best_split], info_gain[best_split])
            node.create_child(~test_res[:, best_split], impurity_l[best_split], prob_l[1, best_split], y_local, 'left')
            node.create_child(test_res[:, best_split], impurity_r[best_split], prob_r[1, best_split], y_local, 'right')

            successful_split = True

        return successful_split


## Parallel training helper - used to train trees in parallel
def train_forest_helper(t_id, X, Y, extracted_from, params, seed):
    print 'tree', t_id, X.shape[0], Y.shape[0]
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
            self.trees.extend(Parallel(n_jobs=self.params.njobs)
                (delayed(train_forest_helper)(t_id, X, Y, extracted_from, self.params, seeds[t_id])
                                             for t_id in range(self.params.num_trees)))
        else:
            #print 'Standard training'
            for t_id in range(self.params.num_trees):
                print 'tree', t_id
                tree = Tree(t_id, self.params)
                tree.train(X, Y, extracted_from)
                self.trees.append(tree)
        #print 'num trees ', len(self.trees)

    def test(self, X, max_depth=np.inf):
        if np.any(np.isnan(X)):
            raise Exception('nans should not be present in test X')

        # return the medoid id at each leaf
        op = np.zeros((X.shape[0], len(self.trees)))
        for tt, tree in enumerate(self.trees):
            op[:, tt] = tree.test(X, max_depth)
        return op

    def delete_trees(self):
        del self.trees[:]

    def calc_importance(self):
        imp = [tree.calc_importance() for tree in self.trees]
        return np.vstack(imp).mean(axis=0)
