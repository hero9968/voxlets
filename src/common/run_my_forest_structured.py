import random_forest_structured as srf
import random_forest_structured_old as srf_old
import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.datasets import load_iris
from sklearn.datasets import load_digits
import pickle

if __name__ == '__main__':

    dataset_type = 'digits'  # iris or digits
    if dataset_type == 'iris':
        data = load_iris()
    elif dataset_type == 'digits':
        data = load_digits()  # bigger dataset

    print dataset_type, ' dataset loaded'

    X = data.data
    Y = data.target
    train_size = Y.shape[0] / 2

    #shuffle
    inds = np.random.permutation(Y.shape[0])
    X = X[inds, :]
    Y = Y[inds]

    x_train = X[:train_size, :]
    x_test = X[train_size:, :]
    y_train = Y[:train_size]
    y_test = Y[train_size:]

    # set up structured labels
    num_classes = np.unique(Y).shape[0]
    y_st = np.zeros((Y.shape[0], num_classes))
    y_st[np.arange(Y.shape[0]), Y.astype('int')] = 1.0
    y_st_train = y_st[:train_size, :].copy()
    #y_st_train += np.random.random(y_st_train.shape)-0.5  # add some noise to training
    y_st_test = y_st[train_size:, :]


    print '\nOMA structured forest'

    # RF params
    forest_params = srf.ForestParams()
    forest = srf.Forest(forest_params)

    tic = time.time()
    forest.train(x_train, y_st_train)
    toc = time.time()
    print 'train time', toc-tic

    tic = time.time()
    y_test_pre_ids = forest.test(x_test)
    toc = time.time()
    print 'test time', toc-tic

    # need to look up ids in the training Y space
    # here we just take the id with the biggest vote
    train_set_class = y_train[y_test_pre_ids.astype('int')]
    pre_labels = np.zeros(y_test.shape[0])
    for ii in range(y_test.shape[0]):
        pre_labels[ii] = np.bincount(train_set_class[ii, :]).argmax()

    tes_acc = (pre_labels == y_test).mean()
    print 'OMA structured forest acc ', tes_acc


    # print '\nOMA forest - save and load'
    # file_for = open('forest.obj', 'w')
    # pickle.dump(forest, file_for)
    # file_for.close()
    #
    # file_for_load = open('forest.obj', 'r')
    # forest_load = pickle.load(file_for_load)
    # file_for_load.close()

    # need to look up ids in the training Y space
    # here we just take the id with the biggest vote
    # train_set_class = y_train[y_test_pre_ids.astype('int')]
    # pre_labels = np.zeros(y_test.shape[0])
    # for ii in range(y_test.shape[0]):
    #     pre_labels[ii] = np.bincount(train_set_class[ii, :]).argmax()
    # tes_acc = (pre_labels == y_test).mean()
    # print 'OMA loaded forest acc ', tes_acc


    print '\nsklearn'
    forest_sc = ExtraTreesClassifier(n_estimators=forest_params.num_trees)
    tic = time.time()
    forest_sc = forest_sc.fit(x_train, y_train)
    toc = time.time()
    print 'train time', toc-tic

    tic = time.time()
    skl_y_test = forest_sc.predict(x_test)
    toc = time.time()
    print 'test time', toc-tic

    skl_test_acc = (skl_y_test == y_test).mean()
    print 'forest acc ', skl_test_acc

