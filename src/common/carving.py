'''
classes for carving and fusion of voxel grids.
Typically will be given a voxel grid and an RGBD 'video', and will do the fusion/carving
'''
import numpy as np

class Carver(object):
    '''
    class for voxel carving
    Possible todos:
    - Allow for only a subset of frames to be used
    - Allow for use of TSDF
    '''

    def __init__(self):
        pass


    def set_video(self, video_in):
        self.video = video_in


    def set_voxel_grid(self, voxel_grid):
        self.voxel_grid = voxel_grid


    def carve(self):
        '''
        for each camera, project voxel grid into camera 
        and see which ahead/behind of depth image.
        Use this to carve out empty voxels from grid
        '''
        for count, im in enumerate(self.video.frames):
            print im.frame_id
            print "\nFrame number %d with name %s" % (count, im.frame_id)
         
            # Projecting voxels into image
            xyz = self.voxel_grid.world_meshgrid()
            projected_voxels = im.cam.project_points(xyz)

            # now work out which voxels are in front of or behind the depth image
            # and location in camera image of each voxel
            uv = np.round(projected_voxels[:, :2]).astype(int)
            inside_image = np.logical_and.reduce((uv[:, 0] >= 0,
                                                  uv[:, 1] >= 0,
                                                  uv[:, 1] < im.depth.shape[0], 
                                                  uv[:, 0] < im.depth.shape[1]))
            all_observed_depths = im.depth[uv[inside_image, 1], uv[inside_image, 0]]

            print "%f%% of voxels projected into image" % \
                (float(np.sum(inside_image)) / float(inside_image.shape[0]))

            # doing the voxel carving (bit of a hack to get the correct voxel idxs...)
            known_empty = all_observed_depths > projected_voxels[inside_image, 2]
            known_empty_global_idx = np.where(inside_image)[0][known_empty]
            known_empty_global_sub = np.array(np.unravel_index(known_empty_global_idx,
                                                    self.voxel_grid.V.shape))

            current_vals = self.voxel_grid.get_idxs(known_empty_global_sub.T)   
            self.voxel_grid.set_idxs(known_empty_global_sub.T, current_vals + 1)

            print "%f%% of voxels seen to be empty" % \
                (float(np.sum(known_empty)) / float(known_empty.shape[0]))


        return self.voxel_grid