'''
here use the saved scale dictionary to copy and resize the meshed objects
'''
import os, sys
import cPickle as pickle
import shutil
import numpy as np

sys.path.append(os.path.expanduser('~/projects/shape_sharing/src/'))
from common import mesh

data_dir = '/Volumes/HDD/data/others_data/mesh_datasets/databaseFull/'

local_mesh_dir = os.path.expanduser('~/projects/shape_sharing/data/meshes2/')
dict_save_path = local_mesh_dir+ 'scales.pkl'

max_object_size = 0.5

# saving dictionary
scale_dict = pickle.load(open(dict_save_path, 'rb'))

# now copying all the files we have scales for into a new directory
obj_source_template = data_dir + 'models/%s.obj'
obj_dest_template = local_mesh_dir + 'models/%s.obj'

for object_name, object_scale in scale_dict.iteritems():

    print obj_source_template % object_name
    print obj_dest_template % object_name

    ms = mesh.Mesh()
    ms.read_from_obj(obj_source_template % object_name)
    ms.centre_mesh()
    ms.scale_mesh(object_scale / 100) # object scale seems to be in centimeters?

    # after scaling, now check if the object is too big or just the right size actually
    print "Max size is " + str(np.max(ms.range()))
    if np.max(ms.range()) < max_object_size:
        ms.write_to_obj(obj_dest_template % object_name)

    print "Done " + object_name