# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>



import cv, cv2
import numpy as np
import matplotlib.pyplot as plt
import scipy.io


# <codecell>

name = "sponza_bunny2"
path = "../../data/reflections/" + name + "/"
imsize = 1024
nbFrames = 30
res = 128

# <codecell>

origin = np.array([[-0.475, 3.062, -0.582, 1]]).T
#origin = np.array([[0, 0, 0, 1]]).T
volume = np.zeros((res,res,res), dtype=np.uint8)
size = 1.5


coords = np.ones((4,res,res,res))
coords[0:3,:,:,:] = np.mgrid[origin[0]-size:origin[0]+size:res*1j, origin[1]-size:origin[1]+size:res*1j, origin[2]-size:origin[2]+size:res*1j]

X = coords.reshape((4, res**3))

for i in range(nbFrames):

    img_raw = cv2.imread(path + "images/" + "image_" + str(i+1).zfill(3) + ".png", -1)
    img_raw = cv2.resize(img_raw, (imsize, imsize), interpolation = cv.CV_INTER_CUBIC)
    mask = np.uint8(img_raw[:,:,3] > 128)
    fmask = mask.flatten()
        
    mats = scipy.io.loadmat(path + "mats/mat_" + str(i+1) + ".mat")
    K = mats["K"]
    R = mats["R"]
    
    K[0,0] *= -1
    
    R1 = np.linalg.pinv(R).T
    R1 = R1[0:3,0:4]
    P = np.dot(K,R1)
    
    
    p = np.dot(P, X)
    lmbd = p[2,:]
    p = p[0:2,:] / lmbd + np.array([[0.5, 0.5]]).T
    
    ip = np.int32(np.round(p))
    
    valid = np.logical_and(ip[0,:] >= 0, np.logical_and(ip[0,:] < imsize, np.logical_and(ip[1,:] >= 0, ip[1,:] < imsize)))
    valid_volume_coords = np.where(valid)[0]
    valid_img_coords = ip[:,valid_volume_coords]
       
    
    vals = mask[valid_img_coords[1,:], valid_img_coords[0,:]]
    
    tempv = np.zeros(res**3, dtype=np.uint8)
    tempv[valid_volume_coords] = vals
    tempv = np.reshape(tempv, (res, res, res))
    
    volume = volume + tempv
    print "image " + str(i) + " done."
    
print "done."

# <codecell>

mlab.pipeline.volume(mlab.pipeline.scalar_field(volume), vmin=29)
mlab.pipeline.image_plane_widget(mlab.pipeline.scalar_field(volume), plane_orientation='y_axes', slice_index=1,)
mlab.pipeline.image_plane_widget(mlab.pipeline.scalar_field(volume),plane_orientation='z_axes',slice_index=32,)
mlab.outline()
mlab.show()

# <codecell>

d = dict(vol=volume, coords=coords, res=res, origin=origin, size=size)
scipy.io.savemat(path + "vol.mat", d)


