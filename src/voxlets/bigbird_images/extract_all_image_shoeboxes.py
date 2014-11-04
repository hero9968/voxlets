'''computes a shoebox for many points from bigbird images'''
import numpy as np
import cPickle as pickle
import sys, os
sys.path.append(os.path.expanduser('~/projects/shape_sharing/src/'))
sys.path.append('..')
#import scipy.io

from common import voxel_data
from common import paths
from common import mesh
from common import images
from common import features

from shoebox_helpers import *

'''PARAMETERS'''
overwrite = False

number_points_from_each_image = 100

print "Loading in the dictionary"
shoebox_kmeans_path = paths.base_path + "voxlets/dict/clusters_from_images.pkl"
km = pickle.load(open(shoebox_kmeans_path, 'rb'))


''' start of main loop'''
for modelname in paths.modelnames:

    # creating folder if it doesn't exist
    savefolder = paths.base_path + "voxlets/bigbird/%s/" % modelname
    if not os.path.exists(savefolder):
        os.mkdir(savefolder)

    # skipping this model if it seems we've already saved all the required files       
    if num_files_in_dir(savefolder) == 75 and not overwrite:
        print "Skipping model " + modelname
        continue

    # loading the voxel grid for this model
    print "Loading voxel data for " + modelname
    vgrid = voxel_data.BigBirdVoxels()
    vgrid.load_bigbird(modelname)

    # loop over the different images of this object:
    for view in paths.views:

        savepath = savefolder + "%s_classified.mat" % view

        # see if should skip
        if os.path.exists(savepath) and not overwrite:
            print "Skipping " + savepath
            continue

        # load the image and the camera
        im = images.CroppedRGBD()
        im.load_bigbird_from_mat(modelname, view)
        #try:
        #except:
        #    print "Could not load bigbird!"
        #    print "Skipping " + modelname + " " + view
        #    continue

        # sample points from the image mask
        mask = ~np.isnan(im.frontrender)
        sampled_points = random_sample_from_mask(mask, number_points_from_each_image)
        try:

            # now compute the shoeboxes
            im_shoeboxes = shoeboxes_from_image(im, vgrid, sampled_points)

            # now compute the features
            cobweb, spider = features_from_image(im, sampled_points)
        except:
            print "Could not get features!"
            print "Skipping " + modelname + " " + view
            continue

        # save these to a file
        cobweb = np.array(cobweb)
        spider = np.array(spider)
        idxs = km.predict(im_shoeboxes)
        D = dict(idxs=idxs, cobweb=cobweb, spider=spider, sampled_points=sampled_points)
        scipy.io.savemat(savepath, D, do_compression=False)
        #scipy.io.savemat(savepath, D, do_compression=True)
        #pickle.dump(D, open(savepath, 'wb'))

        print "Done " + savepath

    print "Done model " + modelname

