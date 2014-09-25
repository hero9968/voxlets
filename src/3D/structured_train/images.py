'''
classes etc for dealing with depth images
'''
import matplotlib.pyplot as plt
import numpy as np
import scipy.misc
import h5py
import struct
from bitarray import bitarray

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

    def compute_edges(self, method='simple'):
        ''' computes edges in some manner... using some method...'''

        if method=='simple':

            local_depth = np.copy(self.depth)
            local_depth[np.isnan(local_depth)] = 0

            # get the gradient and threshold
            dx,dy = np.gradient(local_depth, 1)
            self.grad_mag = np.sqrt(dx**2 + dy**2)
            self.edges = np.array(np.sqrt(dx**2 + dy**2) > 500.1)
            self.gradient = np.rad2deg(np.arctan2(dy, dx))


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

loadim()

#for b in bits(open(mask_path, 'r')):
 #   print b

#im.load_depth_from_img(depth_path)
#im.compute_edges()
#plt.imshow(im.edges)
#plt.colorbar()
#plt.show()



        #self.edges = compute_edges()





