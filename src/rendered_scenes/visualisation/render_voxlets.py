import subprocess as sp
import sys
import os
sys.path.append(os.path.expanduser('~/projects/shape_sharing/src/'))
from common import paths

# run the voxlets to mesh scriupt by importing it
import voxlets_to_mesh

savedir = paths.RenderedData.voxlets_dictionary_path + '/visualisation/kmeans/'
if not os.path.exists(savedir):
    os.makedirs(savedir)

sp.call([
    paths.blender_path,
    "voxlet_render_quick.blend",
    "-b",
    "-P",
    "blender_voxlet_script.py"],
    stdout=open(os.devnull, 'w'),
    close_fds=True)
