from __future__ import print_function

# from pylab import arange, plot, sin, ginput, show
import numpy as np
import matplotlib.pyplot as plt

H, W = 200, 300
N = 50

# generating data
main_im = np.random.rand(H, W)
points = np.random.rand(N, 2)
points[:, 0] *= W
points[:, 1] *= H

voxlet_ims = [np.random.rand(20, 20) for _ in range(N)]

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