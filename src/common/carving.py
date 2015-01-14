'''
classes for carving and fusion of voxel grids.
Typically will be given a voxel grid and an RGBD 'video', and will do the fusion/carving
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
            b) For each voxel that does, the u, v, position of the voxel in the image
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

        for count, im in enumerate(self.video.frames):

            print "\nFrame number %d with name %s" % (count, im.frame_id)
         
            # now work out which voxels are in front of or behind the depth image
            # and location in camera image of each voxel
            inside_image, uv, depth_to_voxels = self.project_voxels(im)
            print uv.shape
            print inside_image.shape
            print np.sum(inside_image)
            all_observed_depths = im.depth[uv[:, 1], uv[:, 0]]

            print "%f%% of voxels projected into image" % \
                (float(np.sum(inside_image)) / float(inside_image.shape[0]))

            # doing the voxel carving
            known_empty = all_observed_depths > depth_to_voxels
            known_empty_global = np.zeros(self.voxel_grid.V.flatten().shape, dtype=bool)
            known_empty_global[inside_image] = known_empty

            existing_values = self.voxel_grid.get_indicated_voxels(known_empty_global)
            self.voxel_grid.set_indicated_voxels(known_empty_global, existing_values + 1)

            print "%f%% of voxels seen to be empty" % \
                (float(np.sum(known_empty)) / float(known_empty.shape[0]))

        return self.voxel_grid




class Fusion(VoxelAccumulator):
    '''
    fuses images with known camera poses into one tsdf volume
    largely uses kinect fusion algorithm (ismar2011), with some changes and simplifications
    Note that even ismar2011 do not use bilateral filtering in the fusion stage, 
    see just before section 3.4.
    '''

    def truncate(self, x, truncation):
        '''
        truncates values in array x to +/i mu
        '''
        x[x > truncation] = truncation
        x[x < -truncation] = -truncation
        return x


    def fuse(self, mu=0.03):
        '''
        mu is the truncation parameter. 
        Default 0.03 as this is what PCL kinfu uses (measured in m)
        Variables ending in _f are full
            i.e. the same size as the full voxel grid
        Variables ending in _s are subsets
            i.e. typically of the same size as the number of valid voxels
        '''

        # create a 'weights' matrix  (see ismar2011 eqn 11 etc)
        weights_f = self.voxel_grid.blank_copy()
        
        for count, im in enumerate(self.video.frames):

            print "\nFrame number %d with name %s" % (count, im.frame_id)

            # this will store the TSDF from this image only - i.e this is F_{R_k}
            temp_volume_f = self.voxel_grid.blank_copy()
            temp_weights_f = self.voxel_grid.blank_copy()
         
            # now work out which voxels are in front of or behind the depth image
            # and location in camera image of each voxel
            inside_image_f, uv_s, depth_to_voxels_s = self.project_voxels(im)
            all_observed_depths_s = im.depth[uv_s[:, 1], uv_s[:, 0]]

            # Distance between depth image and each voxel perpendicular to the camera origin ray (this 
            # is *not* how kinfu does it: see ismar2011 eqn 6&7 for the real method, 
            # which operates along the camera rays!)
            surface_to_voxel_distance_s = depth_to_voxels_s - all_observed_depths_s
            truncated_distance_s = -self.truncate(surface_to_voxel_distance_s, mu)

            # finding indices of voxels which can be updated, according to eqn 9 and the 
            # text after eqn 12
            valid_voxels_s = surface_to_voxel_distance_s >= -mu # i.e. voxels we learn about from this image

            # could do some weight scaling here...
            temp_weights_f.set_indicated_voxels(inside_image_f, valid_voxels_s)
            temp_volume_f.set_indicated_voxels(inside_image_f, truncated_distance_s)

            # updating the global weights and volume (eqn 11 and 12)
            # Assuming static scenes so do not use eqn 13.
            numerator = ((self.voxel_grid.V * weights_f.V) + (temp_volume_f.V * temp_weights_f.V))
            denom = (weights_f.V + temp_weights_f.V)
            valid_voxels_reshaped_f = temp_weights_f.V.reshape(self.voxel_grid.V.shape) == 1
            self.voxel_grid.V[valid_voxels_reshaped_f] = numerator[valid_voxels_reshaped_f] / denom[valid_voxels_reshaped_f]
            weights_f.V += temp_weights_f.V

        return self.voxel_grid
