'''
classes etc for dealing with depth images
'''
import matplotlib.pyplot as plt
import numpy as np
import scipy.misc
import scipy.io
import scipy.ndimage
import h5py
import struct
from bitarray import bitarray
import cv2

import paths
import mesh
import features

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
        self.depth = np.array(f['depth']).astype(np.float32) / 10000
        self.depth[self.depth==0] = np.nan
        self.assert_depth_rgb_equal()

    def load_depth_from_pgm(self, pgm_path):
        ''' the kinfu routine hack i made does pgm like this'''

        # reading header
        f = open(pgm_path, 'r')
        assert(f.readline().strip()=="P2")
        sizes = f.readline().split()
        height = int(sizes[1])
        width = int(sizes[0])
        max_depth = f.readline()

        # pre-allocating
        self.depth = np.zeros((height, width))
        for row_idx, row in enumerate(f):
            for col_idx, col in enumerate(row.strip().split(" ")):
                self.depth[row_idx, col_idx] = float(col)

        self.depth[self.depth==0] = np.nan

    def load_kinect_defaults(self):
        '''
        sets the intrinsics to some kinect default
        '''
        K = np.array([[570.0, 0, 320.0], [0, 570.0, 240.0], [0, 0, 1]])
        self.set_intrinsics(K)


    def assert_depth_rgb_equal(self):
        pass
        #if self.depth.size > 0 and self.rgb.size > 0:
        #    assert(self.rgb.shape[0] == self.depth.shape[0])
        #    assert(self.rgb.shape[1] == self.depth.shape[1])

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

    def compute_edges_and_angles(self, edge_threshold=0.5):
        ''' 
        computes edges in some manner...
        uses a simple method. Real images should overwrite this function
        to use a more complex edge and angle detection
        '''
        temp_depth = np.copy(self.depth)
        temp_depth[np.isnan(temp_depth)] = 10.0
        #import pdb; pdb.set_trace()

        Ix = cv2.Sobel(temp_depth, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=3)
        Iy = cv2.Sobel(temp_depth, ddepth=cv2.CV_32F, dx=0, dy=1, ksize=3)

        self.angles = np.rad2deg(np.arctan2(Iy, Ix))
        self.angles[np.isnan(self.depth)] = np.nan

        self.edges = np.array((Ix**2 + Iy**2) > edge_threshold**2)
        self.edges = scipy.ndimage.morphology.binary_dilation(self.edges)


    def set_angles_to_zero(self):
        self.angles *= 0


    def reproject_3d(self):
        '''
        creates an (nxm)x3 matrix of all the 3D locations of the points.
        '''
        h, w = self.depth.shape
        us, vs = np.meshgrid(np.arange(w), np.arange(h))
        x = 1000*np.vstack((us.flatten()*self.depth.flatten(), vs.flatten()*self.depth.flatten(), self.depth.flatten()))

        self.xyz = np.dot(self.inv_K, x)
        return self.xyz


    # def get_uvd(self):
    #     '''
    #     returns (nxm)x3 matrix of all the u, v coordinates and the depth at eac one
    #     '''
    #     h, w = self.depth.shape
    #     us, vs = np.meshgrid(np.arange(w), np.arange(h))
    #     x = np.vstack((us.flatten(),
    #                    vs.flatten(),
    #                    self.depth.flatten()))


    def set_camera(self, cam_in):
        self.cam = cam_in


    def get_world_xyz(self):
        return self.cam.inv_project_points(self.get_uvd())


    def compute_ray_image(self):
        '''
        the ray image is an image where the values represent the 
        distance along the rays, as opposed to perpendicular
        '''
        self.reproject_3d()
        dists = np.sqrt(np.sum(self.xyz**2, axis=0))

        self.ray_image = np.reshape(dists, self.depth.shape)/1000
        return self.ray_image


    def set_intrinsics(self, K):
        self.K = K
        self.inv_K = np.linalg.inv(K)


    def disp_channels(self):
        '''plots both the depth and rgb and mask next to each other'''

        plt.clf()
        plt.subplot(221)
        plt.imshow(self.rgb)
        plt.subplot(222) 
        plt.imshow(self.depth)
        plt.subplot(223) 
        plt.imshow(self.mask)
        plt.subplot(224) 
        if hasattr(self, 'edges') and self.edges:
            plt.imshow(self.edges)
            plt.show()



class MaskedRGBD(RGBDImage):
    '''
    for objects which have been viewed on a turntable
    especially good for bigbird etc.
    '''

    def load_bigbird(self, modelname, viewname):
        image_path = paths.bigbird_folder + modelname + "/"

        rgb_path = image_path + viewname + '.jpg'
        depth_path = image_path + viewname + '.h5'
        mask_path = image_path + "masks/" + viewname + '_mask.pbm'

        self.load_depth_from_h5(depth_path)
        self.load_rgb_from_img(rgb_path) 
        self.load_mask_from_pbm(mask_path)

        # here I should remove points from the mask which are nan
        self.mask[np.isnan(self.depth)] = 0

        self.view_idx = viewname
        self.modelname = modelname

        self.set_focal_length(240.0/(np.tan(np.rad2deg(43.0/2.0))))

        self.camera_id = self.view_idx.split('_')[0]
        self.load_intrinsics()

    def load_intrinsics(self):
        '''
        loads the intrinscis for this camera and this model from the h5 file
        '''
        calib_path = paths.bigbird_folder + self.modelname + "/calibration.h5"
        calib = h5py.File(calib_path, 'r')

        self.set_intrinsics(np.array(calib[self.camera_id +'_depth_K']))


    def load_mask_from_pbm(self, mask_path, scale_factor=[]):
        self.mask = self.read_pbm(mask_path)
        if scale_factor:
            self.mask = scipy.misc.imresize(self.mask, scale_factor)

        #print "Loaded mask of size " + str(self.mask.shape)

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


    def depth_difference(self, index):
        '''no back render so will just return a nan for this...'''
        return np.nan


class CroppedRGBD(RGBDImage):
    '''
    rgbd image loaded from matlab mat file, which is already cropped
    '''

    def load_bigbird_from_mat(self, modelname, viewname):
        '''
        loads all the bigbird data from a mat file
        '''
        mat_path = paths.base_path + 'bigbird_cropped/' + modelname + '/' + viewname + '.mat'
        D = scipy.io.loadmat(mat_path)
        self.rgb = D['rgb']
        self.depth = D['depth']     # this is the depth after smoothing
        self.frontrender = D['front_render'] # rendered from the voxel data
        self.backrender = D['back_render'] # rendered from the voxel data
        self.aabb = D['aabb'].flatten()  # [left, right, top bottom]
        self.set_intrinsics(D['T_K_rgb']) # as the depth has been reprojected into the RGB image, don't need the ir extrinsics or intrinscs
        self.H = D['T_H_rgb'] 
        # reshaping the xyz to be row-major... better I think and less likely to break
        self.mask = ~np.isnan(self.frontrender)#D['mask']

        old_shape = [3, self.mask.shape[1], self.mask.shape[0]]

        # the following line is a hack to convert from matlab row-col order to python
        self.xyz = D['xyz'].T.reshape(old_shape).transpose((0, 2, 1)).reshape((3, -1)).T
        
        # loading the spider features
        spider_path = paths.base_path + 'bigbird_cropped/' + modelname + '/' + viewname + '_spider.mat'
        D = scipy.io.loadmat(spider_path)
        self.spider_channels = D['spider']

        # the following line is a hack to convert from matlab row-col order to python
        self.normals = D['normals'].T.reshape(old_shape).transpose((0, 2, 1)).reshape((3, -1)).T
        #self.normals[:, [0, 1]] = self.normals[:, [1, 0]]

        self.angles = 0*self.depth

        self.view_idx = viewname
        self.modelname = modelname

        # need here to create the xyz points from the image in world (not camera) coordinates...
        R = np.linalg.inv(self.H[:3, :3].T)
        T = self.H[3, :3]
        #self.world_xyz = np.dot(R, (self.xyz - T).T).T# + 

        # world up direction - in world coordinates. 
        # this is not the up dir in camera coordinates...
        self.updir = np.array([0, 0, 1]) 

        # loading the appropriate camera here
        cam = mesh.Camera()
        cam.load_bigbird_matrices(modelname, viewname)
        cam.adjust_intrinsic_scale(0.5) # as the image is half-sized

        self.set_camera(cam)


    def disp_spider_channels(self):

        print self.spider_channels.shape
        for idx in range(self.spider_channels.shape[2]):
            plt.subplot(4, 6, idx+1)
            plt.imshow(self.spider_channels[:, :, idx])
        



    def depth_difference(self, index):
        ''' 
        returns the difference in depth between the front and the back
        renders at the specified (i, j) index
        '''
        return self.backrender[index[0], index[1]] - self.frontrender[index[0], index[1]]


    def get_uvd(self):
        '''
        returns an nx3 numpy array where each row is the (col, row) index
        location in the image, and the depth at that pixel
        '''
        cols = np.arange(self.aabb[0], self.aabb[1]+1)
        rows = np.arange(self.aabb[2], self.aabb[3]+1)

        [all_cols, all_rows] = np.meshgrid(cols, rows)

        return np.vstack((all_cols.flatten(), 
                          all_rows.flatten(), 
                          self.depth.flatten())).T


    def get_world_normals(self):
        return self.cam.inv_transform_normals(self.normals)


    def get_features(self, idxs):
        '''
        get the features from the channels etc
        '''
        ce = features.CobwebEngine(t=5, fixed_patch_size=False)
        ce.set_image(self)
        patch_features = ce.extract_patches(idxs)

        se = features.SpiderEngine(self)
        spider_features = se.compute_spider_features(idxs)
        spider_features = features.replace_nans_with_col_means(spider_features)

        all_features = np.concatenate((patch_features, spider_features), axis=1)
        return all_features



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

#im = MaskedRGBD()
#im.load_bigbird("coffee_mate_french_vanilla", "NP1_150")
#im.disp_channels()
#plt.show()


#loadim()


