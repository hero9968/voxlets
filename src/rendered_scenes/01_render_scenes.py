import subprocess as sp
import sys
import os
sys.path.append(os.path.expanduser('~/projects/shape_sharing/src/'))
from common import parameters
from common import paths

for i in range(parameters.RenderData.scenes_to_render):
    sp.call([
        paths.blender_path,
        "data_generation/data/blank.blend",
        "-b",
        "-P",
        "data_generation/physics.py"])
