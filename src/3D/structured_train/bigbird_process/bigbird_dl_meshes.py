'''
script to download and set up the PROCESSED bigbird files
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
    base_path = "/mnt/scratch/mfirman/data/bigbird/processed"
    tmp_dir = "/mnt/scratch/mfirman/tmp_meshes/"
else:
    base_path = "/Users/Michael/projects/shape_sharing/data/bigbird/processed"
    tmp_dir = "/Users/Michael/tmp_meshes/"

if not os.path.exists(base_path):
    os.mkdir(base_path)

if not os.path.exists(tmp_dir):
    os.mkdir(tmp_dir)


def exists(name):

    print "Looking for " + base_path + name
    return os.path.exists(tmp_dir + name + ".tgz")


def download_mesh(name):
    base_url = "http://rll.berkeley.edu/bigbird/aliases/863afb5e73/export/"
    full_url = base_url + name + "/processed.tgz"

    print "Downloading " + full_url
    urllib.urlretrieve(full_url, filename=tmp_dir + "processed.tgz")


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


def unpack_mesh(name):

    print "Extracting file"
    tar = tarfile.open(tmp_dir + "processed.tgz", 'r')
    tar.extractall(path=tmp_dir)
    tar.close()

    os.rename(tmp_dir + "processed.tgz", tmp_dir + name + ".tgz")

    os.rename(tmp_dir + name + "meshes", base_path + name + "/meshes")
    os.rename(tmp_dir + name + "textured_meshes",
              base_path + name + "/textured_meshes")

names = get_object_names()

for idx, name in enumerate(names):

    try:
        if not exists(name):
            download_mesh(name)
            unpack_mesh(name)

    except BaseException, e:
        print 'Failed to do something: ' + str(e)
        print "Some kind of problem :"

    #if not mesh_exists(name):
    #    get_mesh(name)

    print "Done " + str(idx) + " (" + name + ")"
    break
