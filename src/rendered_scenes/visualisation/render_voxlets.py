import subprocess as sp
import sys
import os
sys.path.append(os.path.expanduser('~/projects/shape_sharing/src/'))
from common import paths

sp.call([
    paths.blender_path,
    "voxlet_render_quick.blend",
    "-b",
    "-P",
    "blender_voxlet_script.py"])
