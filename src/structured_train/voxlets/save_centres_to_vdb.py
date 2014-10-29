'''
this script will load the cluster centres from the sklearn kmeans pickled object
and then save them out to a openvdb file
'''
import sys
sys.path.append('/Users/Michael/projects/shape_sharing/src/structured_train/')
sys.path.append('/Users/Michael/builds/openvdb_etc/openvdb')

import numpy as np 
import cPickle as pickle
import pyopenvdb as vdb
from thickness import paths

shoebox_size = (20, 20, 20)

read_path = paths.base_path + "voxlets/shoebox_dictionary_training.pkl"
write_path = paths.base_path + "voxlets/shoebox_dictionary_training.vdb"

print "Loading"
km = pickle.load(open(read_path, 'rb'))

print "Converting"
all_vdb = []
for cen in km.cluster_centers_:
    grid = vdb.FloatGrid()
    grid.copyFromArray(cen.reshape(shoebox_size))
    all_vdb.append(grid)

print "Saving"
vdb.write(write_path, all_vdb)