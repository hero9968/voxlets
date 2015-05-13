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


def get_dl_mesh_names():
    names = []
    return os.listdir(tmp_dir)
    #return names

names = set(get_object_names())
meshnames = set(get_dl_mesh_names())
T = list(names.difference(meshnames))
print T