paths.data = '~/projects/shape_sharing/data/';
paths.data_2d = [paths.data, '2D_shapes/'];
paths.mpeg = [paths.data_2d, 'MPEG7_CE-Shape-1_Part_B/'];
paths.subset = [paths.data_2d, 'MPEG7_subset/'];

paths.subset_files = [paths.subset, 'filelist.mat'];

paths.rotated = [paths.data_2d, 'rotated/'];
paths.rotated_filename = '%02d_%02d_mask.gif';
paths.rotated_savename = [paths.rotated, paths.rotated_filename];

paths.raytraced = [paths.data_2d, 'raytraced/'];
paths.raytraced_savename = [paths.raytraced, '%02d_%02d_mask.mat'];

% train/test split
paths.split_path = [paths.data_2d, 'split.mat'];
paths.test_data = [paths.data_2d, 'test_data.mat'];
paths.train_data = [paths.data_2d, 'train_data.mat'];

paths.predictions = [paths.data_2d, 'predict/'];
paths.models_path = [paths.data_2d, 'models/'];
paths.structured_predict_model_path = [paths.data_2d, 'models/structured_predict.mat'];
paths.structured_predict_si_model_path = [paths.data_2d, 'models/structured_predict_si.mat'];
paths.gaussian_predict_model_path = [paths.data_2d, 'models/gaussian_predict.mat'];



% params for setting up file lists
params.number_subclasses = 3; % how many subclasses from each shape to use

params.aggregating = true;

% angles to rotate masks
params.n_angles = 32;
temp_angles = linspace(0, 360, params.n_angles+1);
params.angles = temp_angles(1:end-1);
clear temp_angles

% size of output image
%params.im_height = 250;
params.im_width = 150;
params.im_min_height = 250;
%params.scale = 0.5;

params.segment_soup.thresholds = [200, 40:-5:5];
params.segment_soup.nms_width = 3;
params.segment_soup.max_segments = 20;

% some hand-defined prediction models
params.gauss_model.mu = 0;
params.gauss_model.sigma = 10;

params.gauss_model.number_bins = 25;

% feature computation params
params.shape_dist.num_samples = 5000;
params.shape_dist.bin_edges = [0:5:150, inf];
params.shape_dist.si_bin_edges = linspace(0, 1, 20);
params.angle_edges = linspace(-1, 1, 10);

params.icp.outlier_distance = 10;

params.normal_radius = 10;
params.transform_type = 'icp';



% loading data for the predictors

