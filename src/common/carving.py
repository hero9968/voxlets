'''
Classes for carving and fusion of images into voxel grids.
Typically will be given a voxel grid and an RGBD 'video', and will do the
fusion/carving
'''
import numpy as np
import voxel_data
from copy import deepcopy
from skimage.restoration import denoise_bilateral
import scipy.ndimage
import scipy.interpolate
import scipy.io
from time import time


class VoxelAccumulator(object):
    '''
    Base class for kinect fusion and voxel carving
    '''
    def __init__(self):
        pass

    def set_video(self, video_in):
        self.video = video_in

    def set_voxel_grid(self, voxel_grid):
        self.voxel_grid = voxel_grid

    def project_voxels(self, im):
        '''
        projects the voxels into the specified camera.
        returns tuple of:
            a) A binary array of which voxels project into the image
            b) For each voxel that does, its uv position in the image
            c) The depth to each voxel that projects inside the image
        '''

        # Projecting voxels into image
        xyz = self.voxel_grid.world_meshgrid()
        projected_voxels = im.cam.project_points(xyz)

        # seeing which are inside the image or not
        uv = np.round(projected_voxels[:, :2]).astype(int)
        inside_image = np.logical_and.reduce((uv[:, 0] >= 0,
                                              uv[:, 1] >= 0,
                                              uv[:, 1] < im.depth.shape[0],
                                              uv[:, 0] < im.depth.shape[1]))
        uv = uv[inside_image, :]
        depths = projected_voxels[inside_image, 2]
        return (inside_image, uv, depths)


class Carver(VoxelAccumulator):
    '''
    class for voxel carving
    Possible todos:
    - Allow for only a subset of frames to be used
    - Allow for use of TSDF
    '''
    def carve(self, tsdf=False):
        '''
        for each camera, project voxel grid into camera
        and see which ahead/behind of depth image.
        Use this to carve out empty voxels from grid
        'tsdf' being true means that the kinect-fusion esque approach is taken,
        where the voxel grid is populated with a tsdf
        '''
        vox = self.voxel_grid

        for count, im in enumerate(self.video.frames):

            # print "\nFrame number %d with name %s" % (count, im.frame_id)

            # now work out which voxels are in front of or behind the depth
            # image and location in camera image of each voxel
            inside_image, uv, depth_to_voxels = self.project_voxels(im)
            all_observed_depths = depth[uv[:, 1], uv[:, 0]]

            print "%f%% of voxels projected into image" % \
                (float(np.sum(inside_image)) / float(inside_image.shape[0]))

            # doing the voxel carving
            known_empty_s = all_observed_depths > depth_to_voxels
            known_empty_f = np.zeros(vox.V.flatten().shape, dtype=bool)
            known_empty_f[inside_image] = known_empty_s

            existing_values = vox.get_indicated_voxels(known_empty_f)
            vox.set_indicated_voxels(known_empty_f, existing_values + 1)

            print "%f%% of voxels seen to be empty" % \
                (float(np.sum(known_empty_s)) / float(known_empty_s.shape[0]))

        return self.voxel_grid


class KinfuAccumulator(voxel_data.WorldVoxels):
    '''
    An accumulator which can be used for kinect fusion etc
    valid_voxels are voxels which have been observed as empty or lie in the
    narrow band around the surface.
    '''
    def __init__(self, gridsize, dtype=np.float16):
        self.gridsize = gridsize
        self.weights = np.zeros(gridsize, dtype=dtype)
        self.tsdf = np.zeros(gridsize, dtype=dtype)
        self.valid_voxels = np.zeros(gridsize, dtype=bool)

    def initialise_from_partial_grid(self, grid_V):
        """
        initialises the accumulators from a partially filled TSDF grid
        """
        self.tsdf = deepcopy(grid_V)
        self.weights = (~np.isnan(grid_V)).astype(float)
        self.valid_voxels = self.weights > 0

    def update(self, valid_voxels, new_weight_values, new_tsdf_values):
        '''
        updates both the weights and the tsdf with a new set of values
        '''
        assert(np.sum(valid_voxels) == new_weight_values.shape[0])
        assert(np.sum(valid_voxels) == new_tsdf_values.shape[0])
        assert(np.prod(valid_voxels.shape) == np.prod(self.gridsize))

        # update the weights (no temporal smoothing...)
        if np.sum(np.isnan(self.weights)) > 0:
            raise Exception("Found nans in the weights")
        valid_voxels = valid_voxels.reshape(self.gridsize)
        new_weights = self.weights[valid_voxels] + new_weight_values

        # tsdf update is more complex. We can only update the values of the
        # 'valid voxels' as determined by inputs
        valid_weights = self.weights[valid_voxels]
        valid_tsdf = self.tsdf[valid_voxels]
        valid_tsdf[np.isnan(valid_tsdf)] = 0

        numerator1 = (valid_weights * valid_tsdf)
        numerator2 = (new_weight_values * new_tsdf_values)
        self.tsdf[valid_voxels] = (numerator1 + numerator2) / (new_weights)

        # assign the updated weights
        self.weights[valid_voxels] = new_weights

        # update the valid voxels matrix
        self.valid_voxels[valid_voxels] = True

        self.temptemptemp = valid_voxels

    def get_current_tsdf(self):
        '''
        returns the current state of the tsdf
        '''
        temp = self.copy()
        temp.weights = []
        temp.tsdf = []
        temp.valid_voxels = []
        temp.V = self.tsdf
        temp.V[self.valid_voxels == False] = np.nan
        return temp


class Fusion(VoxelAccumulator):
    '''
    Fuses images with known camera poses into one tsdf volume
    Largely uses kinect fusion algorithm (ismar2011), with some changes and
    simplifications.
    Note that even ismar2011 do not use bilateral filtering in the fusion
    stage, see just before section 3.4.
    Uses a 'weights' matrix to keep a rolling average (see ismar2011 eqn 11)
    '''
    def truncate(self, x, truncation):
        '''
        truncates values in array x to +/i mu
        '''
        x[x > truncation] = truncation
        x[x < -truncation] = -truncation
        return x

    def _fill_in_nans(self, depth):
        # a boolean array of (width, height) which False where there are
        # missing values and True where there are valid (non-missing) values
        mask = ~np.isnan(depth)

        # location of valid values
        xym = np.where(mask)

        # location of missing values
        xymis = np.where(~mask)

        # the valid values of the input image
        data0 = np.ravel( depth[mask] )

        # three separate interpolators for the separate color channels
        interp0 = scipy.interpolate.NearestNDInterpolator( xym, data0 )

        # interpolate the whole image, one color channel at a time
        guesses = interp0(xymis) #np.ravel(xymis[0]), np.ravel(xymis[1]))
        depth = deepcopy(depth)
        depth[xymis[0], xymis[1]] = guesses
        return depth

    def _filter_depth(self, depth):
        temp = self._fill_in_nans(depth)
        if temp.max() > 1.0:
            factor = temp.max()
            temp /= factor
        else:
            factor = 1.0
        temp_denoised = \
            denoise_bilateral(temp, sigma_range=30, sigma_spatial=4.5)
        temp_denoised[np.isnan(depth)] = np.nan
        temp_denoised *= factor
        return temp_denoised

    def integrate_image(self, im, mu, mask=None, filtering=False,
            measure_in_frustrum=False, just_narrow_band=False,
            vox_size_threshold=np.sqrt(2)):
        '''
        integrates single image into the current grid
        mask is optional argument, which should be boolean array same size as im
        we will only integrate the pixels where mask is true
        '''
        # print "Fusing frame number %d with name %s" % (count, im.frame_id)

        if filtering:
            depth = self._filter_depth(im.depth)
        else:
            depth = im.depth

        if mask is not None:
            depth = depth.copy()
            depth[~mask] = np.nan

        # work out which voxels are in front of or behind the depth image
        # and location in camera image of each voxel
        inside_image_f, uv_s, depth_to_voxels_s = self.project_voxels(im)
        observed_depths_s = depth[uv_s[:, 1], uv_s[:, 0]]

        # Distance between depth image and each voxel perpendicular to the
        # camera origin ray (this is *not* how kinfu does it: see ismar2011
        # eqn 6&7 for the real method, which operates along camera rays!)
        surface_to_voxel_dist_s = depth_to_voxels_s - observed_depths_s

        # ignoring due to nans in the comparisons
        np.seterr(invalid='ignore')

        # finding indices of voxels which can be legitimately updated...
        if just_narrow_band:
            # ...according to my new made up method, good for object level fusion
            valid_voxels_s = np.abs(surface_to_voxel_dist_s) <= mu
        else:
            # ...according to eqn 9 and the text after eqn 12 (of Kinfu)
            valid_voxels_s = surface_to_voxel_dist_s <= mu

        # truncating the distance
        truncated_distance_s = -self.truncate(surface_to_voxel_dist_s, mu)

        # expanding the valid voxels to be a full grid
        valid_voxels_f = deepcopy(inside_image_f)
        valid_voxels_f[inside_image_f] = valid_voxels_s

        truncated_distance_ss = truncated_distance_s[valid_voxels_s]
        valid_voxels_ss = valid_voxels_s[valid_voxels_s]

        self.accum.update(
            valid_voxels_f, valid_voxels_ss, truncated_distance_ss)

        # now update the visible voxels - the array which says which voxels
        # are on the surface. I would like this to be automatically
        # extracted from the tsdf grid at the end but I failed to get this
        # to work properly (using the VisibleVoxels class below...) so
        # instead I will make it work here.
        inlier_threshold = vox_size_threshold * self.voxel_grid.vox_size
        voxels_visible_in_image_s = \
            np.abs(surface_to_voxel_dist_s) < inlier_threshold

        visible_voxels_f = inside_image_f
        visible_voxels_f[inside_image_f] = voxels_visible_in_image_s

        self.visible_voxels.set_indicated_voxels(visible_voxels_f, 1)

        if measure_in_frustrum:
            temp = inside_image_f.reshape(self.in_frustrum.V.shape)
            self.in_frustrum.V[temp] += 1

    def _set_up(self):
        """
        Set up all the variables for the fusion
        """
        # the kinfu accumulator, which keeps the rolling average
        self.accum = KinfuAccumulator(self.voxel_grid.V.shape)
        self.accum.vox_size = self.voxel_grid.vox_size
        self.accum.R = self.voxel_grid.R
        self.accum.inv_R = self.voxel_grid.inv_R
        self.accum.origin = self.voxel_grid.origin

        # another grid to store which voxels are visible, i.e on the surface
        self.visible_voxels = self.voxel_grid.blank_copy()
        self.visible_voxels.V = self.visible_voxels.V.astype(bool)

    def fuse(self, mu, filtering=False, measure_in_frustrum=False,
            inlier_threshold=np.sqrt(2)):
        '''
        mu is the truncation parameter. Default 0.03 as this is what PCL kinfu
        uses (measured in m).
        Variables ending in _f are full
            i.e. the same size as the full voxel grid
        Variables ending in _s are subsets
            i.e. typically of the same size as the number of voxels which ended
            up in the image
        Todo - should probably incorporate the numpy.ma module
        if filter==True, then each image is bilateral filtered before adding in
        '''
        self._set_up()

        # finally a third grid, which stores how many frustrums each voxel has
        # fallen into
        if measure_in_frustrum:
            self.in_frustrum = self.voxel_grid.blank_copy()
            self.in_frustrum.V = self.in_frustrum.V.astype(np.int16)

        for count, im in enumerate(self.video.frames):
            self.integrate_image(im, mu, filtering=filtering,
                measure_in_frustrum=measure_in_frustrum,
                vox_size_threshold=inlier_threshold)

        return self.accum.get_current_tsdf(), self.visible_voxels
