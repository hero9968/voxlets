# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

#import cv, cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import scipy.io

# <codecell>

plt.ion()

modelpath = "/Users/Michael/projects/shape_sharing/data/3D/basis_models/renders/%s/depth_%d.mat"
modelname = "128ecbc10df5b05d96eaf1340564a4de";
halopath = "/Users/Michael/projects/shape_sharing/data/3D/basis_models/halo/mat_%d.mat"

imheight = 240
imwidth = 320
nbFrames = 42
res = 100  # resolution of the voxel grid

# <codecell>

#origin = np.array([[-0.475, 3.062, -0.582, 1]]).T
origin = np.array([[0, 0, 0, 1]]).T
volume = np.zeros((res,res,res), dtype=np.uint8)
size = 0.5


coords = np.ones((4,res,res,res))
coords[0:3,:,:,:] = np.mgrid[origin[0]-size:origin[0]+size:res*1j, origin[1]-size:origin[1]+size:res*1j, origin[2]-size:origin[2]+size:res*1j]

X = coords.reshape((4, res**3))

for i in range(nbFrames):

    print "Printing frame %d" % i

    depth_mats = scipy.io.loadmat(modelpath % (modelname, i+1))
    img_raw = depth_mats["depth"]
    mask = np.uint8(img_raw < 3 )
    fmask = mask.flatten()
    plt.imshow(mask)
    plt.draw()

	# loading extrinsic (R) and intrinsic (K) parameters
    mat_path = halopath % (i+1)
    mats = scipy.io.loadmat(mat_path)
    K = mats["K"]
    R = mats["R"].newbyteorder('=')
        
    # some fudge for some reason...?
    K[0,0] *= -1

	# combining parameters into one
    R1 = np.linalg.pinv(R).T
    R1 = R1[0:3,0:4]
    P = np.dot(K,R1)
    
    # projecting the voxel coordinates through the matrices into image coordinates
    p = np.dot(P, X)
    lmbd = p[2,:]
    p = p[0:2,:] / lmbd + np.array([[0.5, 0.5]]).T
    ip = np.int32(np.round(p))
    
    # finding the voxels which actually land in the image
    valid = np.logical_and(ip[0,:] >= 0, np.logical_and(ip[0,:] < imwidth, np.logical_and(ip[1,:] >= 0, ip[1,:] < imheight)))
    
    valid_volume_coords = np.where(valid)[0]  # like matlab's find
    valid_img_coords = ip[:,valid_volume_coords]
    
    vals = mask[valid_img_coords[1,:], valid_img_coords[0,:]]
    
    tempv = np.zeros(res**3, dtype=np.uint8)
    tempv[valid_volume_coords] = vals
    tempv = np.reshape(tempv, (res, res, res))
    
    volume = volume + tempv
    
print "Done all frames."

d = dict(vol=volume, coords=coords, res=res, origin=origin, size=size)
savepath = "/Users/Michael/projects/shape_sharing/3D/model_render/clement_carving/temp.mat"
scipy.io.savemat(savepath, d)

# <codecell>

#mlab.pipeline.volume(mlab.pipeline.scalar_field(volume), vmin=29)
#mlab.pipeline.image_plane_widget(mlab.pipeline.scalar_field(volume), plane_orientation='y_axes', slice_index=1,)
#mlab.pipeline.image_plane_widget(mlab.pipeline.scalar_field(volume),plane_orientation='z_axes',slice_index=32,)
#mlab.outline()
#mlab.show()


