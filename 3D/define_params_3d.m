%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Setting up paths
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear paths

paths.data = '/Users/Michael/projects/shape_sharing/data/';
paths.data_3d = [paths.data, '3D/'];

% where the database models live
paths.basis_models.root = [paths.data_3d, 'basis_models/'];
paths.basis_models.originals = [paths.basis_models.root, 'databaseFull/models'];
paths.basis_models.centred = [paths.basis_models.root, 'centred'];
paths.basis_models.raytraced = [paths.basis_models.root, 'raytraced/'];
paths.basis_models.rendered =  [paths.basis_models.root, 'renders/%s/depth_%d.mat'];
paths.basis_models.voxelised = [paths.basis_models.root, 'voxelised/'];
paths.basis_models.halo_path = [paths.basis_models.root, 'halo/'];
paths.basis_models.halo_file = [paths.basis_models.halo_path, 'mat_%d.csv'];
% TODO - normals, feature vectors

% list of all the basis model files
paths.basis_models.filelist = [paths.basis_models.root, 'databaseFull/fields/models.txt'];

% where to save the captured data
paths.raytraced = [paths.data_3d, 'raytraced/'];

% where to save the segmented basis files
paths.segmented = [paths.data_3d, 'segmented/'];
paths.segmented_savename = [paths.segmented, '%02d_%02d_segmented.mat'];


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Defining parameters
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear params

fid = fopen(paths.basis_models.filelist);
[filelist] = textscan(fid, '%s');
fclose(fid);
params.model_filelist = filelist{1};
clear filelist fid ans

% params for the rendering of the model
params.model_render.hemisphere = 0;
params.model_render.level = 1;
params.model_render.radius = 1.5;

% doing train/test split
params.test_split.test_fraction = 0.25;
params.test_split.max_test_images = 50;
params.test_split.max_training_images = 200;

