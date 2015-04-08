import os

data_folder = '/Users/Michael/projects/shape_sharing/data/desks/oisin_1/data/'
converter_path = '/Users/Michael/projects/InfiniTAM_Alt/convertor/voxels_to_ply.py'

scenes = [os.path.join(data_folder,o)
          for o in os.listdir(data_folder)
          if os.path.isdir(os.path.join(data_folder,o))]
