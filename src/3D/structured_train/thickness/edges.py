'''
functions for computing edges of a depth image
will probably have a separate class for computing the edges for synthetic
images compared to real images
'''
import numpy as np

class SynthEdgeEngine(object):
    '''
    computes edges for synthetic images

    '''

    def __init__(self, depthimage):
        self.depthimage = depthimage

    def compute_edges(self):
        '''
        returns a binary image of the edge locations
        '''
        self.mask = ~np.isnan(self.depthimage).astype(int)

        # get the gradient and threshold
        dx,dy = np.gradient(self.mask, 2)
        self.edge_image = np.array(np.sqrt(dx**2 + dy**2) > 0.0)

        return self.edge_image

    def compute_edge_angles(self, window_size):
        '''
        returns an image the same shape as the edge image,
        but with the angle of each edge at each relevant pixel location,
        and nans where edges are not present
        '''
        # function [norms, curve] = normals_radius_2d( XY, radius )
        # compute normals for set of 2D points

        edge_points = np.nonzero(self.edge_image).T

        for edge_point in edge_points:
            extracted_patch = self.extract_aligned_patch(self.edge_image, edge_point[0], edge_point[1], window_size, 0)
            angle = self.angle_from_patch(extracted_patch)

    def angle_from_patch(self, patch):
        '''
        returns the gradient angle from a binary patch
        '''
        xy = np.nonzero(patch)

        # getting normal
        if np.sum(patch) < 3:
            return 0
        else:
            rot, dummy = np.linalg.eig(np.cov(xy))
            print rot, dummy
            raise Exception("h")
            #idx = diag(dummy)==min(diag(dummy))
            #normal = rot(:, idx(1))
            #curve(ii) = min(diag(dummy))

        # choose rotation so it points in direction of edge difference


         

    def extract_aligned_patch(self, img_in, row, col, hww, pad_value=[]):
        top = int(row - hww)
        bottom = int(row + hww + 1)
        left = int(col - hww)
        right = int(col + hww + 1)

        im_patch = img_in[top:bottom, left:right]

        if top < 0 or left < 0 or bottom > img_in.shape[0] or right > img_in.shape[1]:
            if not pad_value:
                raise Exception("Patch out of range and no pad value specified")

            pad_left = int(self.negative_part(left))
            pad_top = int(self.negative_part(top))
            pad_right = int(self.negative_part(img_in.shape[1] - right))
            pad_bottom = int(self.negative_part(img_in.shape[0] - bottom))
            #print "1: " + str(im_patch.shape)

            im_patch = self.pad(im_patch, (pad_top, pad_bottom), (pad_left, pad_right),
                                constant_values=100)
            #print "2: " + str(im_patch.shape)
            im_patch[im_patch==100.0] =pad_value # hack as apparently can't use np.nan as constant value

        if not (im_patch.shape[0] == 2*hww+1):
            print im_patch.shape
            print pad_left, pad_top, pad_right, pad_bottom
            print left, top, right, bottom
            im_patch = np.zeros((2*hww+1, 2*hww+1))
        if not (im_patch.shape[1] == 2*hww+1):
            print im_patch.shape
            print pad_left, pad_top, pad_right, pad_bottom
            print left, top, right, bottom
            im_patch = np.zeros((2*hww+1, 2*hww+1))
            #raise Exception("Bad shape!")

        return np.copy(im_patch)


    def dilate_edge_angles(self):
        a = 3
        return a



class DistTransform(object):
    
    '''
    Computes the distance transforms for an edge image
    '''
    def __init__(self, arg):
        
        self.arg = arg

    def set_edge_im(self, edge_im):
        self.edge_im = edge_im

    #def 
        

import paths
import scipy.io
import matplotlib.pyplot as plt

def load_frontrender(modelname, view_idx):
    fullpath = paths.base_path + 'basis_models/renders/' + modelname + '/depth_' + str(view_idx) + '.mat'
    frontrender = scipy.io.loadmat(fullpath)['depth']
    return frontrender


frontrender = load_frontrender(paths.modelnames[0], 12)
edge_engine = SynthEdgeEngine(frontrender)
edges = edge_engine.compute_edges()
edge_angles = np.ones((100, 100))#edge_engine.compute_edge_angles()

# TODO - extract the pixels on the edge and see what value they take!
edge_xy = np.array(np.nonzero(edges)).T
for xy in edge_xy:
    print frontrender[xy[0], xy[1]]


#plt.subplot(131)
#plt.imshow(frontrender)
#plt.subplot(132)
edges_and_mask = edges + edge_engine.mask
plt.imshow(edges_and_mask[100:150, 75:125], interpolation='None')
#plt.subplot(133)
#plt.imshow(edge_angles)
plt.show()



