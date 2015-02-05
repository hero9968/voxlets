import numpy as np


class Bricks(object):
    '''
    class to represent a bricks representation of a voxel scene
    '''
    def __init__(self):
        pass

    def from_voxel_grid(self, grid_in, brick_side, pad_or_crop='pad'):
        '''
        Divides up a voxel grid into equal size blocks.
        i.e. transforms from 3D to 6D
        We have option to pad or crop the grid as we see fit
        '''

        if pad_or_crop == 'pad':
            brick_grid_shape = np.ceil(
                np.array(grid_in.shape).astype(float) / brick_side).astype(int)
            cropped_grid = self._pad(grid_in, brick_grid_shape * brick_side)

        elif pad_or_crop == 'crop':
            brick_grid_shape = np.floor(
                np.array(grid_in.shape).astype(float) / brick_side).astype(int)
            cropped_grid = self._crop(grid_in, brick_grid_shape * brick_side)

        intermediate = np.array([brick_grid_shape[0], brick_side,
                                 brick_grid_shape[1], brick_side,
                                 brick_grid_shape[2], brick_side])

        brick_grid = \
            cropped_grid.reshape(intermediate).transpose((0, 2, 4, 1, 3, 5))

        self.B = brick_grid
        self.brick_grid_shape = brick_grid_shape
        self.original_shape = grid_in.shape
        self.brick_side = brick_side
        self.creation_mode = pad_or_crop

    def to_voxel_grid(self):
        '''
        Transforms a 6D brick grid into a 3D voxel grid, of the original size
        '''
        # transforming grid shape to be a multiple of brick_side...
        print self.brick_grid_shape
        print self.B.shape
        v_grid = self.B.transpose(
            (0, 3, 1, 4, 2, 5)).reshape(
            self.brick_grid_shape * self.brick_side)

        if self.creation_mode == 'pad':
            '''we will crop!'''
            v_grid = self._crop(v_grid, self.original_shape)

        elif self.creation_mode == 'crop':
            ''' we will pad!:'''
            v_grid = self._pad(v_grid, self.original_shape)

        return v_grid

    def to_flat(self):
        '''
        returns a flattened (2D) version of the brick grid, where each row is a
        brick and each column is an equivalent voxel in the brick
        '''
        num_bricks = self.B.shape[0] * self.B.shape[1] * self.B.shape[2]
        vox_per_brick = self.B.shape[3] * self.B.shape[4] * self.B.shape[5]
        assert vox_per_brick == self.brick_side**3
        return self.B.reshape((num_bricks, vox_per_brick))

    def from_flat(self, flat_bricks):
        '''
        takes a flattened version of a brick grid, and transforms into the
        required 6D shape. Uses the built-in parameters for brick side and
        grid size etc
        '''
        brick_shape = self.brick_side * np.ones((3,))
        intermediate_shape = np.concatenate(
            (self.brick_grid_shape, brick_shape), axis=0)
        self.B = flat_bricks.reshape(intermediate_shape)

    def _crop(self, to_crop, desired_shape):
        return to_crop[:desired_shape[0],
                       :desired_shape[1],
                       :desired_shape[2]]

    def _pad(self, to_pad, desired_shape):
        current = np.array(to_pad.shape)
        pad_size = np.array(desired_shape) - current

        pad_size2 = [[0, pad_size[0]],
                     [0, pad_size[1]],
                     [0, pad_size[2]]]

        return np.pad(to_pad, pad_size2, mode='edge')

    def apply_func(self, func):
        '''
        applys function func to each brick, returns an array of all the outputs
        '''
        pass

    def get_adjacient_bricks(self, dir):
        '''
        return a list of all the adjacient pairs of voxels in the direction
        specified
        e.g dir = [0, 0, 1]
        '''
        # loop over each voxel
        base_elements = []
        transformed_elements = []

        for i in range(self.brick_grid_shape[0]):
            for j in range(self.brick_grid_shape[1]):
                for k in range(self.brick_grid_shape[2]):
                    offset_i = i + dir[0]
                    offset_j = j + dir[1]
                    offset_k = k + dir[2]
                    if (offset_i < self.brick_grid_shape[0] and
                            offset_j < self.brick_grid_shape[1] and
                            offset_k < self.brick_grid_shape[2]):
                            base_elements.append(self.B[i, j, k])
                            transformed_elements.append(
                                self.B[offset_i, offset_j, offset_k])
#                           print self.B[i, j, k].shape

        print len(base_elements)
        print len(transformed_elements)
        return base_elements, transformed_elements

        # offset by dir
