import socket
import os

# per-view data paths
host_name = socket.gethostname()
if host_name == 'biryani':
    base_path = '/media/ssd/data/'
else:
    base_path = os.path.expanduser("~/projects/shape_sharing/data/")

model = base_path + '/rendered_arrangements/voxlets/models/oma.pkl'


directory = base_path + 'aaron'
temp_dirs = [os.path.split(x[0])[1] for x in os.walk(directory)][1:]
test_sequence = [{
    'folder': base_path + '/aaron/',
    'scene': temp_dir,
    'frames':[0],
    'name': temp_dir}
    for temp_dir in temp_dirs]

voxlet_prediction_img_path = base_path + "/aaron/%s/predictions/%s.png"
voxlet_prediction_folderpath = base_path + "/aaron/%s/predictions/"



