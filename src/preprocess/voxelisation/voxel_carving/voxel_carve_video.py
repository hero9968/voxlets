'''
script to voxel carve a scene given a video of depth frames
in the long run this video should be very short, just with filepaths and paramters
the meat should be in another place

in the short run this will probably not be the case

TODO:
- TSDF fusion instead of straight voxel carving?
- GPU for speed? (probably too much!)
'''
import sys, os
import numpy as np
import scipy
import cPickle as pickle
import time
import hickle

sys.path.append(os.path.expanduser('~/projects/shape_sharing/src/'))
from common import images
from common import voxel_data
from common import carving
from common import paths

# doing a loop here to loop over all possible files...
for scenename in paths.rendered_primitive_scenes[:1]:

    input_data_path = paths.scenes_location + scenename
    pose_filename = 'poses.yaml'

    vid = images.RGBDVideo()
    vid.load_from_yaml(input_data_path, pose_filename)
    #vid.play()

    # initialise voxel grid (could add a helper function to make this more explicit...?)
    vox = voxel_data.WorldVoxels()
    vox.V = np.zeros((150, 150, 75), np.float32)
    vox.set_voxel_size(0.01)
    vox.set_origin((-0.75, -0.75, 0))

    carver = carving.Carver()
    carver.set_video(vid.subvid([0, 1]))
    carver.set_voxel_grid(vox)
    vox = carver.carve()

#    print vox.V.shape
 #   vox.V = np.random.rand(vox.V.shape[0], vox.V.shape[1], vox.V.shape[2]).astype(vox.V.dtype)
  #  print vox.V.shape

    vox._clear_cache()

    print dir(vox)

    # save here! (how? frame, grid...?)
    A = time.clock()
    savepath = paths.scenes_location + scenename + '/voxelgrid.mat'
    print "Saving to %s" % savepath
    D = dict(grid=vox.V)
    scipy.io.savemat(savepath, D)

    print "Took %f seconds" % (time.clock() - A)
    A = time.clock()
    
    savepath = paths.scenes_location + scenename + '/voxelgrid.mat2'
    print "Saving to %s" % savepath
    D = dict(grid=vox.V)
    scipy.io.savemat(savepath, D, do_compression=True)

    print "Took %f seconds" % (time.clock() - A)
    A = time.clock()

    savepath = paths.scenes_location + scenename + '/voxelgrid.pkl'
    print "Saving to %s" % savepath
    with open(savepath, 'wb') as f:
        pickle.dump(vox, f)

    print "Took %f seconds" % (time.clock() - A)
    A = time.clock()

    savepath = paths.scenes_location + scenename + '/voxelgrid.npy'
    print "Saving to %s" % savepath
    with open(savepath, 'wb') as f:
        np.save(f, vox.V)
    
    print "Took %f seconds" % (time.clock() - A)
    A = time.clock()

    savepath = paths.scenes_location + scenename + '/voxelgrid.npz'
    print "Saving to %s" % savepath
    with open(savepath, 'wb') as f:
        np.savez(f, vox.V)
    
    print "Took %f seconds" % (time.clock() - A)
    A = time.clock()

    with open(savepath, 'rb') as f:
        T = np.load(f)
        
    print "Loading took %f seconds" % (time.clock() - A)
    A = time.clock()

    savepath = paths.scenes_location + scenename + '/voxelgrid.npz2'
    print "Saving to %s" % savepath
    with open(savepath, 'wb') as f:
        np.savez_compressed(f, vox.V)

    print "Took %f seconds" % (time.clock() - A)
    A = time.clock()

    with open(savepath, 'rb') as f:
        T = np.load(f)
        
    print "Loading took %f seconds" % (time.clock() - A)
    A = time.clock()

    savepath = paths.scenes_location + scenename + '/voxelgrid.npz3'
    print "Saving to %s" % savepath
    with open(savepath, 'wb') as f:
        np.savez_compressed(f, vox)

    print "Took %f seconds" % (time.clock() - A)
    A = time.clock()
    with open(savepath, 'rb') as f:
        T = np.load(f)


    print "Took %f seconds" % (time.clock() - A)
    A = time.clock()

    print "Saving using custom routine"
    savepath = paths.scenes_location + scenename + '/voxelgrid_custom.pkl'
    vox.save(savepath)
    print "Took %f seconds" % (time.clock() - A)

    A = time.clock()
    vox2 = voxel_data.load_voxels(savepath)
    print vox.V.flatten()[:100]
    print vox2.V.flatten()[:100]
    assert(np.all(vox2.V == vox.V))
    
    print "Loading took %f seconds" % (time.clock() - A)