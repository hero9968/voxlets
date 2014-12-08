'''
idea is to load a depth image, and for each pixel work out the dictionary item which belongs to it
the ultimate aim is to be able to create a training set based on the dictionary
'''
import sys
sys.path.append('/Users/Michael/projects/shape_sharing/src/')
from common import paths

import shutil

basepath = '/Volumes/BlackBackup/shape_sharing_backups/shape_sharing/data/voxlets/bigbird/troll_predictions/'

types = ['bb', 'oma', 'modal'] #'just_cobweb'

views = ['nutrigrain_harvest_blueberry_bliss_NP1_0', 
            'pop_secret_butter_NP3_216',
            'pringles_bbq_NP2_0',
            'progresso_new_england_clam_chowder_NP3_312',
            'south_beach_good_to_go_peanut_butter_NP1_312']

for t in types:
    for v in views:

        original_path = basepath + t + '/' + v + '.mat'
        new_path = './data/' + t + '/'

        shutil.copy(original_path, new_path)

        original_path = basepath + t + '/' + v + '.png'
        new_path = './data/' + t + '/'

        shutil.copy(original_path, new_path)