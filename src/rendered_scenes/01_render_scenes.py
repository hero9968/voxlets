import subprocess as sp
import sys
sys.path.append('/Users/Michael/projects/shape_sharing/src/')
from common import parameters
from common import paths

for i in range(parameters.General.scenes_to_render):
    sp.call([
        paths.blender_path,
        "spinaround/spin.blend",
        "data_generation/data/blank.blend",
        "-b",
        "-P",
        "data_generation/physics.py"])
