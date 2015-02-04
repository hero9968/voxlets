import numpy as np


def divide_up_voxel_grid(grid_in, brick_side):
    '''
    Divides up a voxel grid into equal size blocks.
    If grid is not exact multiple of the brick size, then it shaves off a bit
    of the grid
    '''
    brick_grid_shape = np.floor(
        np.array(grid_in.shape) / brick_side).astype(int)
    brick_shape = np.array([brick_side, brick_side, brick_side])

    full_grid_shape = np.concatenate((brick_grid_shape, brick_shape), axis=0)

    brick_grid = np.zeros(full_grid_shape)

    # doing the dividing... should use as_strided but for now...
    for i_idx in range(full_grid_shape[0]):
        i = i_idx * brick_side
        for j_idx in range(full_grid_shape[1]):
            j = j_idx * brick_side
            for k_idx in range(full_grid_shape[2]):
                k = k_idx * brick_side
                temp = grid_in[i:i+brick_side, j:j+brick_side, k:k+brick_side]
                brick_grid[i_idx, j_idx, k_idx, :, :, :] = temp

    return brick_grid


def flatten_brick_grid(brick_grid_in):
    '''
    takes a NxMxPxIxJxK 6D array, and returns a (N*M*P)x(I*J*K) 2D array
    '''
    N = np.prod(brick_grid_in.shape[:3])
    M = np.prod(brick_grid_in.shape[3:])
    return brick_grid_in.reshape((N, M))


def reform_voxel_grid_from_flat_bricks(
    flat_bricks, brick_grid_shape, brick_side, original_shape=None):
    '''
    Divides up a voxel grid into equal size blocks.
    If grid is not exact multiple of the brick size, then it shaves off a bit
    of the grid
    original_shape is the desired shape of the output array - this function
    will crop or pad the array as needed to achieve the original_shape
    '''

    # transforming grid shape to be a multiple of brick_side...

    brick_shape = np.array([brick_side, brick_side, brick_side])
    intermediate_shape = np.concatenate((brick_grid_shape, brick_shape), axis=0)

    print "intermediate is ", intermediate_shape

    intermediate = flat_bricks.reshape(intermediate_shape)

    full_grid_shape = np.array(brick_grid_shape) * brick_side
    brick_grid = np.zeros(full_grid_shape)

    # doing the dividing... should use as_strided but for now...
    for i_idx in range(brick_grid_shape[0]):
        i = i_idx * brick_side
        for j_idx in range(brick_grid_shape[1]):
            j = j_idx * brick_side
            for k_idx in range(brick_grid_shape[2]):
                k = k_idx * brick_side
                temp = intermediate[i_idx, j_idx, k_idx, :, :, :]
                brick_grid[i:i+brick_side, j:j+brick_side, k:k+brick_side] = temp

    if original_shape:
        extra = original_shape[0] - brick_grid.shape[0]
        if extra < 0:
            brick_grid = brick_grid[:original_shape[0], :, :]
        elif extra > 0:
            brick_grid = np.pad(brick_grid, [[0, extra], [0, 0], [0, 0]],
                'edge')

        extra = original_shape[1] - brick_grid.shape[1]
        if extra < 0:
            brick_grid = brick_grid[:, :original_shape[1], :]
        elif extra > 0:
            brick_grid = np.pad(brick_grid, [[0, 0], [0, extra], [0, 0]],
                'edge')

        extra = original_shape[2] - brick_grid.shape[2]
        if extra < 0:
            brick_grid = brick_grid[:, :, :original_shape[2]]
        elif extra > 0:
            brick_grid = np.pad(brick_grid, [[0, 0], [0, 0], [0, extra]],
                'edge')

    return brick_grid

