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
paths.basis_models.fv_path = [paths.basis_models.root, 'fv/'];
paths.basis_models.fv_file = [paths.basis_models.fv_path, '%s.mat'];
paths.basis_models.halo_file = [paths.basis_models.halo_path, 'mat_%d.csv'];
paths.basis_models.halo_file_mat = [paths.basis_models.halo_path, 'mat_%d.mat'];
% TODO - normals, feature vectors

% list of all the basis model files
paths.basis_models.filelist = [paths.basis_models.root, 'databaseFull/fields/models.txt'];

% where to save the captured data
paths.raytraced = [paths.data_3d, 'raytraced/'];

% where to save the segmented basis files
paths.segmented = [paths.data_3d, 'segmented/'];
paths.segmented_savename = [paths.segmented, '%02d_%02d_segmented.mat'];

paths.shape_dist_dict = [paths.basis_models.root, 'shape_dist_dict.mat'];

paths.structured_model_file = [paths.data_3d, 'structured_model.mat'];

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

% segmentation parameters
params.segment_soup.min_cluster_size = 500;
params.segment_soup.max_cluster_size = 1e6;
params.segment_soup.num_neighbours = 50;
params.segment_soup.smoothness_threshold = (7.0 / 180.0) * pi;
params.segment_soup.curvature_threshold = 1.0;
params.segment_soup.overlap_threshold = 0.2; % higher number = fewer clusters

% parameters for the shape distribution
params.shape_dist.num_samples = 20000;
load(paths.shape_dist_dict, 'dict')
params.shape_dist.dict = dict;
clear dict

% kinect parameters
params.focal_length = 240/(tand(43/2));
% note that old focal length was 570.3 - make sure to use a suitable one
params.half_intrinsics = [params.focal_length/2, 0, 160; ...
                          0, params.focal_length/2, 120; ...
                          0, 0, 1];

params.full_intrinsics = [params.focal_length, 0, 320; ...
                          0, params.focal_length, 240; ...
                          0, 0, 1];
                       
params.rgbddataset_intrinsics = [570.3, 0, 320; ...
                          0, 570.3, 240; ...
                          0, 0, 1];