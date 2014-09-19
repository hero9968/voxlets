import numpy as np
from sklearn.ensemble import RandomForestRegressor


class ClassSampledForest(object):
    '''
    extends the sklean forest to add some additional functionality
    '''

    def __init__(self, n_estimators, n_jobs, max_depth):

        self.n_estimators = n_estimators
        self.n_jobs = n_jobs
        self.max_depth = max_depth
        

    def sample_classes(self, all_class_labels, fraction_to_sample):

        # get vector of unique class labels
        unique_class_labels, class_idxs = np.unique(all_class_labels, return_inverse=True)
        unique_class_idxs = np.unique(class_idxs)
        
        # choose how many class labels we want to sample
        number_to_sample = int(fraction_to_sample * len(unique_class_labels))
        
        # sample (with replacement) the classes we want to use in our sample
        classes_to_use = np.random.choice(unique_class_idxs, size=number_to_sample, replace=True)
        print "Classes to use are: " + str(classes_to_use)

        final_choice = np.zeros(class_idxs.shape)
        for idx, this_class in enumerate(classes_to_use)
            final_choice = np.boolean_or(class_idxs == this_class, final_choice)
        
        print "Final choice is " + str(final_choice.shape)
        #final_choice = [True if t in classes_to_use else False for t in all_class_labels]

        print "Using " + str(np.sum(final_choice)) + " items, out of " + str(len(final_choice))
        
        # find which of our original vector is in the selected sample
        return final_choice


    def fit(self, X, Y, class_Y):
        '''
        does the fitting like the normal forest
        however, samples items at a class level
        model.fit(X,Y)
        '''

        self.estimators_ = []

        for i in range(self.n_estimators):

            # sample from the data
            print "Before sampling:"
            print "X is " + str(X.shape)
            print "Y is " + str(Y.shape)

            to_use = self.sample_classes(class_Y, 0.66)
            temp_X = X[to_use, :]
            temp_Y = Y[to_use]

            print "After sampling:"
            print "temp_X is " + str(temp_X.shape)
            print "temp_Y is " + str(temp_Y.shape)

            tree = RandomForestRegressor(n_estimators=1, n_jobs=self.n_jobs, max_depth=self.max_depth)
            tree.fit(temp_X, temp_Y)
            self.estimators_.append(model)

    def predict(self, X, aggregator='median'):
        '''
        Does prediction but with some additonal options like doing per tree
        '''

        tree_results = [tree.predict(X) for tree in self.estimators_]

        if aggregator == 'median':
            avg_results = np.median(tree_results, axis=0)
        elif aggregator == 'mean':
            avg_results = np.mean(tree_results, axis=0)
        else:
            raise Exception("Unknown aggregator")

        return avg_results, tree_results




#all_classes = [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6, 7, 7, 7, 8, 8, 8]
#print np.array(sample_classes(all_classes, 0.7)).astype(int)

