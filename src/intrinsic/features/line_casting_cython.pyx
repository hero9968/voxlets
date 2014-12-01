# python setup.py build_ext --inplace

import numpy as np
cimport numpy as np

def inner_loop(np.ndarray[double, ndim=2] input_im, 
                int start_row,
                int start_col,
                np.ndarray[int, ndim=1] direction,
                np.ndarray[int, ndim=2] output_im):
    '''
    given a start pixel and a direction, head in that direction until you run out of 
    space on the input image
    modifies the output image, so returns None
    '''

    cdef unsigned int i, j, H, W
    cdef int cumsum

    i = start_row
    j = start_col
    H = input_im.shape[0]
    W = input_im.shape[1]

    cumsum = -1

    # doing full check is unnecessary for certain directions - this could be optimised
    while i >= 0 and j >= 0 and i < H and j < W:

        # this is the test which is used to decide when to start counting
        if input_im[i, j] >= 0:
            cumsum = 0
        else: #if cumsum != -1:
            cumsum += 1

        output_im[i, j] = cumsum

        i += direction[0]
        j += direction[1]


def outer_loop(np.ndarray[double, ndim=2] input_im, np.ndarray[int, ndim=1] direction):
    '''
    given an input image and a direction in which to go, does all of the iterations
    '''
    
    cdef unsigned int H, W
    H = input_im.shape[0]
    W = input_im.shape[1]
    
    # must initalise the output image, as in cython
    cdef np.ndarray output_im = np.zeros([H, W], np.int32)


    ##################################################
    if direction[0] > 0:
        # start at all locations on the top
        for col in xrange(W):
            inner_loop(input_im, 0, col, direction, output_im)
        
    elif direction[0] < 0:
        # start at all locations on the bottom
        for col in xrange(W):
            inner_loop(input_im, H-1, col, direction, output_im)


    ##################################################
    if direction[1] > 0:
        # start at all locations on left edge
        for row in xrange(H):
            inner_loop(input_im, row, 0, direction, output_im)

    elif direction[1] < 0:
        # start at all locations on right edge
        for row in xrange(H):
            inner_loop(input_im, row, W-1, direction, output_im)

    return output_im



