from __future__ import print_function

'''
Improvements:
1. Better plotting of the voxlets
    - 3D rendering Better
    - floor plane?
    -
2. Top down view...
3. Use depth image
    - Show which region of depth image falls into voxlet?

'''

# from pylab import arange, plot, sin, ginput, show
import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import imread

base_path = '/media/ssd/data/oisin_house/predictions/different_data_split/saved_00233_[134]/'

# generating data
main_im = imread(base_path + 'input.png')
H, W, _ = main_im.shape
points = np.loadtxt(open(base_path + 'sampled_locations.txt'), delimiter=',').astype(int)[:, ::-1]
N = points.shape[0]

voxlet_ims = [imread(base_path + 'voxlets/compiled_%03d.png' % f) for f in range(N)]

# initial plotting
plt.subplot(121)
plt.imshow(main_im)
plt.plot(points[:, 0], points[:, 1], 'bo')
plt.axis('off')

while True:

    x = plt.ginput(1)[0]
    print("Clicked ", x)

    # finding nearest point
    dists = np.linalg.norm(points - x, axis=1)
    min_idx = np.argmin(dists)

    # updating plots
    plt.subplot(121)
    plt.plot(points[:, 0], points[:, 1], 'bo')
    plt.plot(points[min_idx, 0], points[min_idx, 1], 'ro')
    plt.title("clicked (%f, %f)" % (x[0], x[1]))

    plt.subplot(122)
    plt.imshow(voxlet_ims[min_idx])
    plt.axis('off')

    plt.draw()