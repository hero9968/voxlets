import random_forest as rf
import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris

if __name__ == '__main__':

    # RF params
    forest_params = rf.ForestParams()
    forest = rf.Forest(forest_params)

    print 'load data'
    # just binary for now
    data = load_iris()
    X = data.data[:100, :]
    Y = data.target[:100]
    train_size = Y.shape[0] / 2

    #shuffle
    inds = np.random.permutation(Y.shape[0])
    X = X[inds, :]
    Y = Y[inds]

    x_train = X[:train_size, :]
    x_test = X[train_size:, :]
    y_train = Y[:train_size]
    y_test = Y[train_size:]

    print x_train.shape
    print x_test.shape


    print '\nOMA forest'
    tic = time.time()
    forest.train(x_train, y_train)
    toc = time.time()
    print 'train time', toc-tic

    y_test_pre = forest.test(x_test)
    tes_acc = ((y_test_pre > 0.5) == y_test).mean()
    print 'OMA forest acc ', tes_acc


    print '\nscikit-learn'
    forest = RandomForestClassifier(n_estimators=forest_params.num_trees)
    tic = time.time()
    forest = forest.fit(x_train, y_train)
    toc = time.time()
    print 'scikit train time', toc-tic

    skl_y_test = forest.predict_proba(x_test)[:, 1]
    skl_test_acc = ((skl_y_test > 0.5) == y_test).mean()
    print 'scikit forest acc ', skl_test_acc
