'''
script to download and set up the bigbird files
'''
import os
import urllib
import requests
from bs4 import BeautifulSoup
import tarfile

import socket

# per-view data paths
host_name = socket.gethostname()
if host_name == 'troll':
    base_path = "/mnt/scratch/mfirman/data/bigbird"
    tmp_dir = "/mnt/scratch/mfirman/tmp/"
else:
    base_path = "/Users/Michael/projects/shape_sharing/data/bigbird/"
    tmp_dir = "/Users/Michael/tmp/"



filetypes = [["/", ".jpg"], 
            ["/", ".h5"], 
            ["/masks/", "_mask.pbm"]]

unwanted_views = []
for elevation in [1, 2, 3, 4, 5]:
    for azimuth in range(0, 360, 3):
        if (azimuth%24) != 0:
            unwanted_views.append("NP" + str(elevation) + "_" + str(azimuth))

def remove_unwanted_rgbd(objname):
    '''removes about 75% of the views'''
    for filetype in filetypes:
        for unwanted_view in unwanted_views:
            fullname = base_path + objname + filetype[0] + unwanted_view + filetype[1]

            if os.path.exists(fullname):
                os.remove(fullname)

def num_files(foldername, extension):
    '''returns the number of files with extension in the folder foldername'''
    return len([name for name in os.listdir(foldername) 
        if os.path.isfile(os.path.join(foldername, name))
        and os.path.splitext(name)[1] == extension])

def is_decimated(objname):
    '''checks to see if a named object has already been decimated'''
    depth_file_count =  num_files(base_path + objname, '.h5') - 1
    rgb_file_count =  num_files(base_path + objname, '.jpg')
    mask_file_count =  num_files(base_path + objname + "/masks", '.pbm')
    print depth_file_count, rgb_file_count, mask_file_count

    return depth_file_count == 75 \
        and rgb_file_count == 75 \
        and mask_file_count == 75

def exists(name):
    print "Looking for " + base_path + name
    return os.path.exists(tmp_dir + name + ".tgz")

def download_rgbd(name):
    base_url = "http://rll.berkeley.edu/bigbird/aliases/863afb5e73/export/"
    full_url = base_url + name + "/rgbd.tgz"
    
    print "Downloading " + full_url
    urllib.urlretrieve(full_url, filename=tmp_dir + "rgbd.tgz")

def get_object_names():
    '''getting all the object names from the web page'''
    r = requests.get("http://rll.berkeley.edu/bigbird/aliases/863afb5e73/")
    data = r.text
    soup = BeautifulSoup(data)

    names = []
    for tdd in soup.find_all('td'):
        if 'class' in tdd.attrs and tdd.attrs['class'][0] == 'name_cell':
            names.append(tdd.contents[0].strip())
    return names

def unpack_rgbd(name):

    print "Extracting file"
    tar = tarfile.open(tmp_dir + "rgbd.tgz", 'r')
    tar.extractall(path=base_path)
    tar.close()

    os.rename(tmp_dir + "rgbd.tgz", tmp_dir + name + ".tgz")

names = get_object_names()
#print names

for idx, name in enumerate(names):

    try:
        if not exists(name):
            download_rgbd(name)
            unpack_rgbd(name)
            print "Downloaded " + name
        else:
            print name + " already exists"

        if not is_decimated(name):
            remove_unwanted_rgbd(name)
            print "Decimated " + name
            is_decimated(name) # should print confirmation
        else:
            print name + " is already decimated"

    except BaseException, e:
        print 'Failed to do something: ' + str(e)
        print "Some kind of problem :"

    #if not mesh_exists(name):
    #    get_mesh(name)

    print "Done " + str(idx) + " (" + name + ")"
