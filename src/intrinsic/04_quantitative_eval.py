import numpy as np
import scipy.io
import sklearn.metrics
import sys, os
sys.path.append(os.path.expanduser('~/projects/shape_sharing/src/'))

from common import paths

# now doing the acu and fp and tp etc as we might as well
# now do for all the data on this machine!
all_pres = []
all_rec = []
all_auc = []

intrinsic_save_path = paths.base_path + '/implicit/bigbird/predictions/%s_%s.mat'

# now will use this prediction to reconstruct one of the original objects from an image
for modelname in paths.test_names:

    for this_view_idx in [0, 10, 20, 30, 40]:

        test_view = paths.views[this_view_idx]        
        
        # saving
        "Loading result to disk"
        loadpath = intrinsic_save_path % (modelname, test_view)
        D = scipy.io.loadmat(loadpath)

        pred = 1 - ((D['prediction'] + 0.03) / 0.06)
        pred = pred[:, :, 15:].flatten()

        gt = D['gt'][:, :, 15:].flatten()

        "now evaluate"
        prescision = sklearn.metrics.precision_score(gt.round().astype(int), pred.round().astype(int))
        recall = sklearn.metrics.recall_score(gt.round().astype(int), pred.round().astype(int))
        print prescision, recall
        auc = sklearn.metrics.roc_auc_score(gt.flatten(), pred)

        all_rec.append(recall)
        all_pres.append(prescision)
        all_auc.append(auc)

    print modelname + ":"    
    print np.array(all_auc)
    print np.mean(np.array(all_auc))
    print np.mean(np.array(all_pres))
    print np.mean(np.array(all_rec))


print "---------------"
print "Final results:"
print np.mean(np.array(all_auc))
print np.mean(np.array(all_pres))
print np.mean(np.array(all_rec))
