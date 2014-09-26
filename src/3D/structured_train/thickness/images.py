'''
classes etc for dealing with depth images
'''
import matplotlib.pyplot as plt
import numpy as np
import scipy.misc
import scipy.io
import h5py
import struct
from bitarray import bitarray
import cv2

import paths

class RGBDImage(object):

    def __init__(self):
        self.rgb = np.array([])
        self.depth = np.array([])
        self.focal_length = []

    def load_rgb_from_img(self, rgb_path, scale_factor=[]):

        self.rgb = scipy.misc.imread(rgb_path)
        if scale_factor:
            self.rgb = scipy.misc.imresize(self.rgb, scale_factor)
        assert(self.rgb.shape[2] == 3)
        self.assert_depth_rgb_equal()

    def load_depth_from_img(self, depth_path):

        self.depth = scipy.misc.imread(depth_path)
        self.assert_depth_rgb_equal()
    
    def load_depth_from_h5(self, depth_path):

        f = h5py.File(depth_path, 'r') 
        self.depth = np.array(f['depth'])
        self.assert_depth_rgb_equal()

    def assert_depth_rgb_equal(self):

        if self.depth.size > 0 and self.rgb.size > 0:
            assert(self.rgb.shape[0] == self.depth.shape[0])
            assert(self.rgb.shape[1] == self.depth.shape[1])

    def set_focal_length(self, focal_length):
        self.focal_length = focal_length

    def disp_channels(self):
        '''plots both the depth and rgb next to each other'''
        plt.clf()
        plt.subplot(121)
        plt.imshow(self.rgb)
        plt.subplot(122) 
        plt.imshow(self.depth)
        plt.show()

    def print_info(self):
        '''prints info about the thing'''

        if self.rgb.size > 0:
            print "RGB image has shape: " + str(self.rgb.shape)
        else:
            print "No RGB image present"

        if self.depth.size > 0:
            print "Depth image has shape: " + str(self.depth.shape)
        else:
            print "No Depth image present"

        if hasattr(self, 'mask'):
            print "Mask has shape: " + str(self.mask.shape)

        print "Focal length is " + str(self.focal_length)

    def compute_edges_and_angles(self):
        ''' 
        computes edges in some manner...
        uses a simple method. Real images should overwrite this function
        to use a more complex edge and angle detection
        '''
        temp_depth = np.copy(self.depth)
        temp_depth[np.isnan(temp_depth)] = 10.0

        Ix = cv2.Sobel(temp_depth, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=3)
        Iy = cv2.Sobel(temp_depth, ddepth=cv2.CV_32F, dx=0, dy=1, ksize=3)

        self.angles = np.rad2deg(np.arctan2(Iy, Ix))
        self.angles[np.isnan(self.depth)] = np.nan

        self.edges = np.array(Ix**2 + Iy**2 > 0.5)

    def set_angles_to_zero(self):
        self.angles *= 0


class MaskedRGBD(RGBDImage):
    '''
    for objects which have been viewed on a turntable
    especially good for bigbird etc.
    '''

    def load_mask_from_pbm(self, mask_path, scale_factor=[]):
        self.mask = self.read_pbm(mask_path)
        if scale_factor:
            self.mask = scipy.misc.imresize(self.mask, scale_factor)

        print "Loaded mask of size " + str(self.mask.shape)

    def read_pbm(self, fname):
        '''
        reads a pbm image. not tested in the general case but works on the masks
        '''
        with open(fname) as f:
            data = [x for x in f if not x.startswith('#')] #remove comments

        header = data.pop(0).split()
        dimensions = [int(header[2]), int(header[1])]

        arr = np.fromstring(data.pop(0), dtype=np.uint8)
        return np.unpackbits(arr).reshape(dimensions)


    def disp_channels(self):
        '''plots both the depth and rgb and mask next to each other'''

        plt.clf()
        plt.subplot(131)
        plt.imshow(self.rgb)
        plt.subplot(132) 
        plt.imshow(self.depth)
        plt.subplot(133) 
        plt.imshow(self.mask)
        plt.show()



class CADRender(RGBDImage):
    '''
    class representing a CAD render model
    perhaps this should inherit from a base rgbd/image class... perhaps not
    '''

    def __init__(self):
        '''init with the parent class, but also add the backdepth'''
        RGBDImage.__init__(self)
        self.backdepth = np.array([])

    def load_from_cad_set(self, modelname, view_idx):
        '''loads models from the cad training set'''
        self.modelname = modelname
        self.view_idx = view_idx

        self.depth = self.load_frontrender(modelname, view_idx)
        self.backdepth = self.load_backrender(modelname, view_idx)
        self.mask = ~np.isnan(self.depth)
        #self.mask = self.extract_mask(frontrender)

        self.focal_length = 240.0/(np.tan(np.rad2deg(43.0/2.0))) / 2.0

    def load_frontrender(self, modelname, view_idx):
        fullpath = paths.base_path + 'basis_models/renders/' + modelname + '/depth_' + str(view_idx) + '.mat'
        return scipy.io.loadmat(fullpath)['depth']

    def load_backrender(self, modelname, view_idx):
        fullpath = paths.base_path + 'basis_models/render_backface/' + modelname + '/depth_' + str(view_idx) + '.mat'
        return scipy.io.loadmat(fullpath)['depth']

    def depth_difference(self, index):
        ''' 
        returns the difference in depth between the front and the back
        renders at the specified (i, j) index
        '''
        return self.backdepth[index[0], index[1]] - self.depth[index[0], index[1]]




#image_path = '/Users/Michael/data/rgbd_scenes2/rgbd-scenes-v2/imgs/scene_01/'
#image_name = '00401'

def loadim():
    image_path = "/Users/Michael/projects/shape_sharing/data/bigbird/coffee_mate_french_vanilla/"
    image_name = "NP1_150"

    rgb_path = image_path + image_name + '.jpg'
    depth_path = image_path + image_name + '.h5'
    mask_path = image_path + "masks/" + image_name + '_mask.pbm'

    im = MaskedRGBD()
    im.load_depth_from_h5(depth_path)
    im.load_rgb_from_img(rgb_path, (480, 640))
    im.load_mask_from_pbm(mask_path, (480, 640))
    im.print_info()
    im.disp_channels()

def loadcadim():
    im = CADRender()
    im.load_from_cad_set(paths.modelnames[30], 30)
    im.compute_edges_and_angles()

    plt.clf()
    plt.subplot(121)
    plt.imshow(im.angles, interpolation='nearest')
    plt.colorbar()
    plt.subplot(122)
    plt.imshow(np.isnan(im.angles).astype(int) - np.isnan(im.depth).astype(int), interpolation='nearest')
    plt.colorbar()

    plt.show()
    im.print_info()

#loadcadim()

#loadim()


