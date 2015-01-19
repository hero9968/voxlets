'''
Classes for carving and fusion of images into voxel grids.
Typically will be given a voxel grid and an RGBD 'video', and will do the
fusion/carving
'''
import numpy as np
import voxel_data


class VoxelAccumulator(object):
    '''
    base class for kinect fusion and voxel carving
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

            print "\nFrame number %d with name %s" % (count, im.frame_id)

            # now work out which voxels are in front of or behind the depth
            # image and location in camera image of each voxel
            inside_image, uv, depth_to_voxels = self.project_voxels(im)
            all_observed_depths = im.depth[uv[:, 1], uv[:, 0]]

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

    def update(self, valid_voxels, weight_values, tsdf_values):
        '''
        updates both the weights and the tsdf with a new set of values
        '''
        assert(np.sum(valid_voxels) == weight_values.shape[0])
        assert(np.sum(valid_voxels) == tsdf_values.shape[0])
        assert(np.prod(valid_voxels.shape) == np.prod(self.gridsize))

        # update the weights (no temporal smoothing...)
        valid_voxels = valid_voxels.reshape(self.gridsize)
        new_weights = self.weights[valid_voxels] + weight_values

        # tsdf update is more complex. We can only update the values of the
        # 'valid voxels' as determined by inputs
        valid_weights = self.weights[valid_voxels]
        valid_tsdf = self.tsdf[valid_voxels]

        numerator1 = (valid_weights * valid_tsdf)
        numerator2 = (weight_values * tsdf_values)
        self.tsdf[valid_voxels] = (numerator1 + numerator2) / (new_weights)

        # assign the updated weights
        self.weights[valid_voxels] = new_weights

        # update the valid voxels matrix
        self.valid_voxels[valid_voxels] = True

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

    def fuse(self, mu):
        '''
        mu is the truncation parameter. Default 0.03 as this is what PCL kinfu
        uses (measured in m).
        Variables ending in _f are full
            i.e. the same size as the full voxel grid
        Variables ending in _s are subsets
            i.e. typically of the same size as the number of voxels which ended
            up in the image
        Todo - should probably incorporate the numpy.ma module
        '''
        # the accumulator, which keeps the rolling average
        accum = KinfuAccumulator(self.voxel_grid.V.shape)
        accum.vox_size =self.voxel_grid.vox_size
        accum.R =self.voxel_grid.R
        accum.origin =self.voxel_grid.origin

        for count, im in enumerate(self.video.frames):

            print "Fusing frame number %d with name %s" % (count, im.frame_id)

            # work out which voxels are in front of or behind the depth image
            # and location in camera image of each voxel
            inside_image_f, uv_s, depth_to_voxels_s = self.project_voxels(im)
            observed_depths_s = im.depth[uv_s[:, 1], uv_s[:, 0]]

            # Distance between depth image and each voxel perpendicular to the
            # camera origin ray (this is *not* how kinfu does it: see ismar2011
            # eqn 6&7 for the real method, which operates along camera rays!)
            surface_to_voxel_dist_s = depth_to_voxels_s - observed_depths_s

            # finding indices of voxels which can be legitimately updated,
            # according to eqn 9 and the text after eqn 12
            valid_voxels_s = surface_to_voxel_dist_s <= mu

            # truncating the distance
            truncated_distance_s = -self.truncate(surface_to_voxel_dist_s, mu)

            # expanding the valid voxels to be a full grid
            valid_voxels_f = inside_image_f
            valid_voxels_f[inside_image_f] = valid_voxels_s

            truncated_distance_ss = truncated_distance_s[valid_voxels_s]
            valid_voxels_ss = valid_voxels_s[valid_voxels_s]

            accum.update(valid_voxels_f, valid_voxels_ss, truncated_distance_ss)

        return accum.get_current_tsdf()



class VisibleVoxels(object):
    '''
    This class uses a voxel grid to find the visible voxels! This is a new way
    of doing it...
    '''

    def __init__(self):
        pass

    def set_voxel_grid(self, voxel_grid):
        self.voxel_grid = voxel_grid

    def axis_aligned_zero_crossings(self, V, axis):
        diff = np.diff(np.sign(V), axis=axis)

        padding = [[0, 0], [0, 0], [0, 0]]
        padding[axis] = [1, 0]
        diff1 = np.pad(diff, padding, 'edge')

        padding = [[0, 0], [0, 0], [0, 0]]
        padding[axis] = [0, 1]
        diff2 = np.pad(diff, padding, 'edge')

        return np.logical_or(diff1, diff2)

    def find_visible_voxels(self):
        '''
        finds all the zero-crossings in the voxel grid
        must ensure only use zero-crossings in the narrow-band
        '''
        dx = self.axis_aligned_zero_crossings(self.voxel_grid.V, axis=0)
        dy = self.axis_aligned_zero_crossings(self.voxel_grid.V, axis=1)
        dz = self.axis_aligned_zero_crossings(self.voxel_grid.V, axis=2)

        temp = np.logical_or.reduce((dx, dy, dz))
        temp[np.isnan(self.voxel_grid.V)] = 0
        temp[self.voxel_grid.V > 0.02] = 0

        visible_grid = self.voxel_grid.blank_copy()
        visible_grid.V = temp
        return visible_grid


