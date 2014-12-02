# python setup.py build_ext --inplace

import numpy as np
cimport numpy as np

'''
input image has the following three values for pixels:
    1:  Voxels observed to be FULL
    0:  Voxels observed to be EMPTY 
    -1: Voxels with UNKNOWN state
'''


def inner_loop(np.ndarray[double, ndim=2] input_im, 
                int start_row,
                int start_col,
                np.ndarray[int, ndim=1] direction,
                np.ndarray[int, ndim=2] output_im,
                np.ndarray[np.uint8_t, ndim=2] observed_indicator):
    '''
    given a start pixel and a direction, head in that direction until you run out of 
    space on the input image

    This function modifies the output image(s), so returns None

    observed_indicator is an array of the same size as output_im, as stores whether or not the last voxel
    seen was an OBSERVED one
    '''

    cdef unsigned int i, j, H, W
    cdef int cumsum
    cdef int observed_to_be_full

    i = start_row
    j = start_col
    H = input_im.shape[0]
    W = input_im.shape[1]

    cumsum = -1 #special indicator to signify inifinity   np.iinfo(np.int32).max
    observed_to_be_full = 0

    # doing full check is unnecessary for certain directions - this could be optimised
    while i >= 0 and j >= 0 and i < H and j < W:
        # this is the test which is used to decide when to start counting
        if input_im[i, j] == 0:
            # this is voxel known to be empty
            cumsum = 0
            observed_to_be_full = 0

        else:
            if cumsum != -1:
                # voxels either observed to be full OR unknown
                # check for -1 checks whether should be accumulating or not...
                cumsum += 1

            if input_im[i, j] == 1:
                # voxel observed by the camera
                # variable stays true until encounter a voxel known to be empty
                observed_to_be_full = 1


        output_im[i, j] = cumsum
        observed_indicator[i, j] = observed_to_be_full

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
    cdef np.ndarray observed_to_be_full = np.zeros([H, W], np.uint8)


    ##################################################
    if direction[0] > 0:
        # start at all locations on the top
        for col in xrange(W):
            inner_loop(input_im, 0, col, direction, output_im, observed_to_be_full)
        
    elif direction[0] < 0:
        # start at all locations on the bottom
        for col in xrange(W):
            inner_loop(input_im, H-1, col, direction, output_im, observed_to_be_full)


    ##################################################
    if direction[1] > 0:
        # start at all locations on left edge
        for row in xrange(H):
            inner_loop(input_im, row, 0, direction, output_im, observed_to_be_full)

    elif direction[1] < 0:
        # start at all locations on right edge
        for row in xrange(H):
            inner_loop(input_im, row, W-1, direction, output_im, observed_to_be_full)

    return output_im, observed_to_be_full





def inner_loop_3d(np.ndarray[double, ndim=3] input_im, 
                int start_row,
                int start_col,
                int start_slice,
                np.ndarray[int, ndim=1] direction,
                np.ndarray[int, ndim=3] output_im,
                np.ndarray[np.uint8_t, ndim=3] observed_indicator):
    '''
    given a start pixel and a direction, head in that direction until you run out of 
    space on the input image

    This function modifies the output image(s), so returns None

    observed_indicator is an array of the same size as output_im, as stores whether or not the last voxel
    seen was an OBSERVED one
    '''

    cdef unsigned int i, j, k, H, W, D
    cdef int cumsum
    cdef int observed_to_be_full

    i = start_row
    j = start_col
    k = start_slice
    H = input_im.shape[0]
    W = input_im.shape[1]
    D = input_im.shape[2]

    cumsum = -1 #special indicator to signify inifinity   np.iinfo(np.int32).max
    observed_to_be_full = 0

    # doing full check is unnecessary for certain directions - this could be optimised
    while i >= 0 and j >= 0 and k >= 0 and i < H and j < W and k < D:
        # this is the test which is used to decide when to start counting
        if input_im[i, j, k] == 0:
            # this is voxel known to be empty
            cumsum = 0
            observed_to_be_full = 0

        else:
            if cumsum != -1:
                # voxels either observed to be full OR unknown
                # check for -1 checks whether should be accumulating or not...
                cumsum += 1

            if input_im[i, j, k] == 1:
                # voxel observed by the camera
                # variable stays true until encounter a voxel known to be empty
                observed_to_be_full = 1


        output_im[i, j, k] = cumsum
        observed_indicator[i, j, k] = observed_to_be_full

        i += direction[0]
        j += direction[1]
        k += direction[2]




def outer_loop_3d(np.ndarray[double, ndim=3] input_im, np.ndarray[int, ndim=1] direction):
    '''
    given an input image and a direction in which to go, does all of the iterations
    '''

    cdef unsigned int H, W, D
    H = input_im.shape[0]
    W = input_im.shape[1]
    D = input_im.shape[2]

    # must initalise the output image, as we are in cython
    cdef np.ndarray output_im = np.zeros([H, W, D], np.int32)
    cdef np.ndarray observed_to_be_full = np.zeros([H, W, D], np.uint8)


    ##################################################
    if direction[0] > 0:
        # start at all locations on top face
        for col in xrange(W):
            for sslice in xrange(D):
                inner_loop_3d(input_im, 0, col, sslice, direction, output_im, observed_to_be_full)
        
    elif direction[0] < 0:
        # start at all locations on bottom face
        for col in xrange(W):
            for sslice in xrange(D):
                inner_loop(input_im, H-1, col, sslice, direction, output_im, observed_to_be_full)


    ##################################################
    if direction[1] > 0:
        # start at all locations on left face
        for row in xrange(H):
            for sslice in xrange(D):
                inner_loop(input_im, row, 0, sslice, direction, output_im, observed_to_be_full)

    elif direction[1] < 0:
        # start at all locations on right face
        for row in xrange(H):
            for sslice in xrange(D):
                inner_loop(input_im, row, W-1, sslice, direction, output_im, observed_to_be_full)


    ##################################################
    if direction[2] > 0:
        # start at all locations on front face
        for row in xrange(H):
            for col in xrange(W):
                inner_loop(input_im, row, col, 0, direction, output_im, observed_to_be_full)

    elif direction[2] < 0:
        # start at all locations on back face
        for row in xrange(H):
            for col in xrange(W):
                inner_loop(input_im, row, col, D-1, direction, output_im, observed_to_be_full)


    return output_im, observed_to_be_full


