'''
Package for displaying of data and image manipulations etc.
'''
import numpy as np
import compute_data

def nans(shape, dtype=float):
    '''
    creates empty nan image
    '''
    a = np.empty(shape, dtype)
    a.fill(np.nan)
    return a

def reconstruct_image(indices, values, shape):
    '''
    return image of size `shape` with `values` in the positions indicated by `indices`
    '''
    assert(indices.shape[1] == values.shape[0])
    out = nans(shape)
    out[indices[0], indices[1]] = values
    return out

def crop_concatenate(all_ims, padding=0):
    '''
    puts ims in list all_ims next to each other, and crops each of them for better display
    '''
    im_array = np.array(all_ims)
    mask = ~np.any(np.isnan(im_array), axis=0)
    
    left = compute_data.findfirst(np.any(mask, axis=0)) - padding
    right = mask.shape[1] - compute_data.findfirst(np.any(mask, axis=0)[::-1]) + padding
    top = compute_data.findfirst(np.any(mask, axis=1)) - padding
    bottom = mask.shape[0] - compute_data.findfirst(np.any(mask, axis=1)[::-1]) + padding
    
    all_arrays_cropped = [im[top:bottom, left:right] for im in all_ims]
    return np.concatenate(all_arrays_cropped, axis=1)
