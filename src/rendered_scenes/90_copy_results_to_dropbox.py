import sys
import os
sys.path.append(os.path.expanduser("~/projects/shape_sharing/src/"))
from common import paths
import shutil

outpath = '/home/michael/Dropbox/PhD/Projects/Shape_sharing_data/predictions/'


for sequence in paths.RenderedData.test_sequence():

    gen_renderpath = paths.RenderedData.voxlet_prediction_img_path % \
        ('oma_implicit', sequence['name'], '%s')

    fname = 'all_' + sequence['name']

    savename = gen_renderpath.replace('png', 'pdf') % fname

    if os.path.exists(savename):
        shutil.copy2(savename, outpath)
