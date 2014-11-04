'''
train the model dammit
'''
import numpy as np
import cPickle as pickle
import scipy.io
import sys
sys.path.append('/Users/Michael/projects/shape_sharing/src/')
from sklearn.ensemble import RandomForestClassifier

from common import paths

"Parameters"
max_data_in_subsample = 300000
number_trees = 100
if paths.host_name != 'troll':
    small_sample = True
else:
    small_sample = False
max_depth = 15
if small_sample: print "WARNING: Just computing on a small sample"


####################################################################
print "Loading in all the data..."
shoeboxes = []
features = []
for count, modelname in enumerate(paths.train_names):

    # loading the data
    loadpath = paths.bigbird_training_data_mat % modelname
    print "Loading from " + loadpath

    D = scipy.io.loadmat(loadpath)
    shoeboxes.append(D['shoeboxes'])

    features.append(D['features'])
    
    if count > 2 and small_sample:
        print "SMALL SAMPLE: Stopping"
        break

np_all_sboxes = np.concatenate(shoeboxes, axis=0).astype(np.float16)


####################################################################
print "Loading the dictionary"
km = pickle.load(open(paths.voxlet_dict_path, 'rb'))


####################################################################
print "Assigning shoeboxes to clusters"
idx_assign = km.predict(np_all_sboxes)


####################################################################
print "Now training the forest"

np_features = np.concatenate(features, axis=0)
print "Idx assign has shape " + str(idx_assign.shape)

if max_data_in_subsample > np_features.shape[0]:
    print "Using all data..."
    np_features_subset = np_features
else:
    print "Subsampling data..."
    to_use_for_clustering = np.random.randint(0, np_features.shape[0], size=(max_data_in_subsample))
    np_features_subset = np_features[to_use_for_clustering, :]
    idx_assign = idx_assign[to_use_for_clustering]

print "Features shape before subsampling :" + str(np_features.shape)
print "Features shape after subsampling :" + str(np_features_subset.shape)

forest = RandomForestClassifier(n_estimators=number_trees, criterion="entropy", oob_score=True, max_depth=max_depth)
forest.fit(np_features_subset, idx_assign)

print "Done training, now saving"
pickle.dump(forest, open(paths.voxlet_model_path, 'wb'))
