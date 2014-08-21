import os
import numpy as np
import matplotlib.pyplot as plt
#from sklearn.ensemble import RandomForestRegressor
#import cPickle as pickle
import compute_data
import train_model
from mpl_toolkits.mplot3d import Axes3D

def project_depth_to_xyz(depth):
    # stack of homogeneous coordinates of each image cell
    x = np.arange(0, depth.shape[1])
    y = np.arange(0, depth.shape[0])
    [xgrid, ygrid] = np.meshgrid(x, y);
    n = len(xgrid.flatten())
    temp = np.array([xgrid.flatten(), ygrid.flatten(), np.ones((n, 1))])
    full_stack = temp * depth.flatten();
    
    # apply inverse intrinsics, and convert to standard coloum format
    K = np.array([[304.6377, 0, 160], [0, 304.6377, 120.0000], [0, 0, 1]])
    ans = np.linalg.solve(K,full_stack)
    return ans

modelname = '1444822d28a7f2632a526e2e9a7e9ae'#'109d55a137c042f5760315ac3bf2c13e'#'2566f8400d964be69c3cb90632bf17f3' #
view_idx = 12

# loading the features to test on and the ground truth data
all_features = compute_data.features_and_depths(modelname, view_idx, -1)
X = np.array(all_features['patch_features'])
X = train_model.nan_to_value(X, 0)
Y_gt = np.array(all_features['depth_diffs'])
Y_gt = train_model.nan_to_value(Y_gt, 0)

# rf prediction
#Y_pred = clf.predict(X)

# computing error
##error = np.mean(np.abs(Y_pred - Y_gt))
#print "Error is " + str(error)

# reprojecting the prediction to 3D...
frontrender = compute_data.load_frontrender(modelname, view_idx)
xyz = project_depth_to_xyz(frontrender)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(xyz[0], xyz[1], xyz[2])
plt.show()

# showing a 2D slice through the 3D reprojection