
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cPickle as pickle
import sys, os
sys.path.append(os.path.expanduser('~/projects/shape_sharing/src/'))
sys.path.append('../bigbird_images3/')

from common import paths
from common import voxel_data
from common import mesh
from common import images
from common import features
from thickness_regress import combine_data
import reconstructer

folder_path = paths.base_path + 'other_3D/from_biryani/troll_predictions/'
name = 'frame_20141120T185954.959422_'
load_path = folder_path + name + '.pkl'
D = pickle.load(open(load_path, 'rb'))

im = images.RealRGBD()
im_load_path =  paths.base_path + 'other_3D/from_biryani/' + 'mdf/' + name + '.mat'
im.load_from_mat(im_load_path)

import copy

save_folder = paths.base_path + 'other_3D/from_biryani/slices/' + name + '/%s_%4.5d.png'
#os.mkdir(paths.base_path + 'other_3D/from_biryani/slices/' + name + '/')


def band_im(im, height):
    band_width = 0.0025
    height_off_ground_im = im.get_world_xyz()[:,2].reshape(im.mask.shape)
    band_im = np.abs(height_off_ground_im - height) < band_width
    lower_im = height_off_ground_im < (height - band_width)
    temp_rgb = np.copy(im.rgb)
    alpha = 0.7
    temp_rgb[lower_im==1, 0] = (temp_rgb[lower_im==1, 0])*(1-alpha) + (temp_rgb[lower_im==1, 0])*0+1 * alpha
    temp_rgb[band_im==1, 0] = 255
    temp_rgb[band_im==1, 1] = 0
    temp_rgb[band_im==1, 2] = 0
    print im.get_world_xyz().shape
    print band_im.shape
    return temp_rgb, im.get_world_xyz()[band_im.flatten()==1, :]

plt.rcParams['figure.figsize'] = (15.0, 20.0)
cmap = plt.cm.get_cmap('bwr_r')
saving = True

for idx in range(80, D.V.shape[2]):
    
    plt.clf()
    
    # getting height of this slice
    height = D.idx_to_world(np.array([0, 0, idx])[np.newaxis, :]).flatten()[2]
    print height.shape
    
    slice_rgb, slice_xyz = band_im(im, height)
        
    if saving:
        fig = plt.figure(frameon=False)
    else:
        plt.subplot(121)
        
    plt.imshow(slice_rgb)
    plt.gca().axes.get_xaxis().set_visible(False)
    plt.gca().axes.get_yaxis().set_visible(False)
    
    if saving:
        save_name = save_folder % ('rgb', idx)
        fig.savefig(save_name, bbox_inches='tight', pad_inches=0)

    if saving:
        fig = plt.figure(frameon=False)
    else:
        plt.subplot(122)

    plt.imshow(D.V[:, :, idx].T)
    plt.clim([-0.03, 0.03])
    plt.set_cmap(cmap)
    plt.gca().axes.get_xaxis().set_visible(False)
    plt.gca().axes.get_yaxis().set_visible(False)


    
    # now try to get the real-world xyz positions of the observed points on the slice
    idx_locations = D.world_to_idx(slice_xyz)
    plt.hold(True)
    plt.plot(idx_locations[:, 0], idx_locations[:, 1], 'g.')
    plt.hold(False) 
    plt.xlim([0, D.V.shape[0]])
    plt.ylim([0, D.V.shape[1]])
    
    #plt.show() 
    
    if saving:
        save_name = save_folder % ('slice', idx)
        fig.savefig(save_name, bbox_inches='tight', pad_inches=0)

    print "Done " + str(idx)
