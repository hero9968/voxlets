import numpy as np
from thickness import paths
from thickness import images
import scipy.io
rubbish = set()

for modelname in paths.modelnames:
    for view_idx in range(1, 42):
        im = images.CADRender()
        im.load_from_cad_set(modelname, view_idx)
        sum_nonzero = np.sum(~np.isnan(im.depth))
        if sum_nonzero <= 100:
            rubbish.add(modelname)
            print modelname, view_idx, sum_nonzero

print rubbish