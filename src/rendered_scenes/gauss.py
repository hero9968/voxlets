
import numpy as np
from copy import deepcopy
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import FactorAnalysis

class ApproxNN(object):
    '''
    does approximated nearest neighbour
    '''
    def __init__(self, n_dims):
        # n_dims is the number of dimensions to use
        self.n_dims = n_dims

    def set_X(self, X):
        self.X = X

    def predict(self, test_X, known_dims):
        '''currently only predicts for a single example'''
        assert np.all(known_dims.shape == test_X.shape)

        to_use_dims = np.where(known_dims)[0]
        print to_use_dims.shape
        to_use_dims = np.random.choice(to_use_dims, self.n_dims)

        lookup_subset = test_X[to_use_dims]
        training_subset = np.take(self.X, to_use_dims, axis=1)

        print lookup_subset.shape
        print training_subset.shape

        # now do the nn lookup
        nbrs = NearestNeighbors(n_neighbors=10, algorithm='brute')
        nbrs.fit(training_subset)
        distances, indices = nbrs.kneighbors(lookup_subset)

        print distances.shape, indices.shape

        return self.X[indices[0], :]

class GaussImpute(object):
    '''
    aim is to fit a gaussian to training data and use this to fill in missing
    values in a new test dataset
    COULD use svd or similar to 'flatten down' the learned gaussian shape, this
    might help to speed up the prediction or something...
    '''
    def __init__(self):
        pass

    def fit(self, X):
        self.mean = np.mean(X, axis=0)
        self.cov = np.cov(X.T).T

    def predict(self, X, mask):
        '''
        guesses the missing values in X, as defined by the true entries in mask
        '''
        #print np.all(X.shape == mask.shape)
        assert np.all(X.shape == mask.shape)
        assert mask.dtype == np.bool

        X = np.atleast_2d(X)
        mask = np.atleast_2d(mask)

        pred_means, pred_covs = [], []

        for X_row_t, mask_row in zip(X, mask):

            #print mask_row.shape
            #print X_row.shape
            #print self.cov.shape
            X_row = deepcopy(X_row_t)

            # doing the conditional gaussian distribution
            # Notation is as used in the prince book, section 5.5 (p73)

            x_2 = X_row[~mask_row]
            #print mask_row.shape
            #print self.mean.shape
            mu_2 = self.mean[~mask_row]
            mu_1 = self.mean[mask_row]

            cov_11 = deepcopy(self.cov)[mask_row, :][:, mask_row]
            #print cov_11.shape
            cov_21 = deepcopy(self.cov)[~mask_row, :][:, mask_row]
            cov_12 = deepcopy(self.cov)[mask_row, :][:, ~mask_row]
            cov_22 = deepcopy(self.cov)[~mask_row, :][:, ~mask_row]
            #print cov_21.T.shape
            #print cov_22.shape

            cov_21_T_times_cov_22_inv = cov_12.dot(np.linalg.pinv(cov_22))
            #print cov_21_T_times_cov_22_inv.shape

            #print cov_22.shape
            #inv_cov_22 = np.linalg.pinv(cov_22)
            #print inv_cov_22.shape

            temp_mean = mu_1 + cov_21_T_times_cov_22_inv.dot(x_2 - mu_2)
            temp_cov = cov_11 - cov_21_T_times_cov_22_inv.dot(cov_21)

            new_mean = deepcopy(X_row)
            new_mean[mask_row] = temp_mean
            pred_means.append(new_mean)

            #pred_covs.append(new_cov)

        return pred_means, x_2 - mu_2, cov_22 #, pred_covs


class FAGaussImpute(object):
    '''
    aim is to fit a gaussian to training data and use this to fill in missing
    values in a new test dataset
    This one uses factor analysis
    '''
    def __init__(self):
        pass

    def fit(self, X, n_components):
        self.fa = FactorAnalysis(n_components=n_components)
        self.fa.fit(X)

    def predict(self, X, mask):
        '''
        guesses the missing values in X, as defined by the true entries in mask
        '''
        #print np.all(X.shape == mask.shape)
        assert np.all(X.shape == mask.shape)
        assert mask.dtype == np.bool

        X = np.atleast_2d(X)
        mask = np.atleast_2d(mask)

        pred_means, pred_covs = [], []

        for X_row_t, mask_row in zip(X, mask):


            # doing the conditional gaussian distribution
            # Notation is as used in the prince book, section 5.5 (p73)

            X_row = deepcopy(X_row_t)
            x_2 = X_row[~mask_row]

            mu_2 = self.fa.mean_[~mask_row]
            mu_1 = self.fa.mean_[mask_row]

            the1 = self.fa.components_[:, mask_row].T
            the2 = self.fa.components_[:, ~mask_row].T

            noise_1 = np.diag(self.fa.noise_variance_[mask_row])
            noise_2 = np.diag(self.fa.noise_variance_[~mask_row])

            cov_11 = np.dot(the1, the1.T) + noise_1
            cov_21 = np.dot(the2, the1.T)
            cov_22 = np.dot(the2, the2.T) +  noise_2

            cov_21_T_times_cov_22_inv = cov_21.T.dot(np.linalg.inv(cov_22))

            temp_mean = mu_1 + cov_21_T_times_cov_22_inv.dot(x_2 - mu_2)
            temp_cov = cov_11 - cov_21_T_times_cov_22_inv.dot(cov_21)

            new_mean = deepcopy(X_row)
            new_mean[mask_row] = temp_mean
            pred_means.append(new_mean)

            #pred_covs.append(new_cov)

        return pred_means, x_2 - mu_2, cov_22 #, pred_covs

