# from __future__ import print_function

# from pylab import arange, plot, sin, ginput, show
import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import imread
import os, sys
sys.path.append('../')
from common import voxlets, scene
import yaml
train_data = yaml.load(open('/media/ssd/data/oisin_house/train_test/test.yaml'))
params = yaml.load(open('../real_data/oisin_house/training_params.yaml'))

print params
vox_params = params['voxlet_sizes']['short']

sc = scene.Scene(mu=0.025)
sc.load_sequence(train_data[0], frame_nos=0, segment_with_gt=True)


# initial plotting
plt.subplot(121)
plt.imshow(sc.im.rgb)

while True:

    x = plt.ginput(1)[0]
    x = np.round(np.array((x[1], x[0]))).astype(int)
    print "Clicked ", x

    # extracting a voxlet from here
    sc.voxlet_params = vox_params
    vox = sc.extract_single_voxlet(x, 'gt_tsdf')

    # # finding nearest point
    # dists = np.linalg.norm(points - x, axis=1)
    # min_idx = np.argmin(dists)


    plt.subplot(122)
    plt.cla()
    plt.imshow(vox.V[:, :, 10])
    # plt.axis('off')

    plt.draw()