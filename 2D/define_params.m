%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Setting up paths
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear paths

paths.data = '~/projects/shape_sharing/data/';
paths.data_2d = [paths.data, '2D_shapes/'];

% where the raw data files exist
paths.mpeg = [paths.data_2d, 'MPEG7_CE-Shape-1_Part_B/'];
paths.filelist = [paths.data_2d, 'filelist.mat'];

% where to save the raytraced files
paths.raytraced = [paths.data_2d, 'raytraced/'];
paths.raytraced_savename = [paths.raytraced, '%04d_raytraced.mat'];

% where to save the segmented files
paths.segmented = [paths.data_2d, 'segmented/'];
paths.segmented_savename = [paths.segmented, '%02d_%02d_segmented.mat'];

% train/test split
paths.split_path = [paths.data_2d, 'split.mat'];
paths.test_data = [paths.data_2d, 'test_data.mat'];
paths.test_data_subset = [paths.data_2d, 'test_data_subset.mat'];
paths.train_data = [paths.data_2d, 'train_data.mat'];
paths.all_images = [paths.data_2d, 'all_images.mat'];

% where to save the models and the predictions
paths.predictions = [paths.data_2d, 'predict/'];
paths.models_path = [paths.data_2d, 'models/'];
paths.structured_predict_model_path = [paths.data_2d, 'models/structured_predict_large.mat'];
paths.structured_predict_si_model_path = [paths.data_2d, 'models/structured_predict_si_large.mat'];
paths.gaussian_predict_model_path = [paths.data_2d, 'models/gaussian_predict.mat'];


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Defining parameters
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear params

% doing train/test split
params.test_split.test_fraction = 0.25;
params.test_split.max_test_images = 50; %500
params.test_split.max_training_images = 200; %20000

% angles to rotate masks
params.n_angles = 32;
temp_angles = linspace(0, 360, params.n_angles+1);
params.angles = temp_angles(1:end-1);
clear temp_angles

% size of output image
params.scale = 0.5;
params.im_min_height = 250;

% normal computation
params.normal_radius = 10;

% segmentation parameters
params.segment_soup.thresholds = [200, 40:-5:5];
params.segment_soup.nms_width = 3;
params.segment_soup.max_segments = 20;
params.segment_soup.overlap_threshold = 0.9;
params.segment_soup.min_size = 10;

% parameters of the gaussian model
params.gauss_model.number_bins = 25;

% feature computation params
params.shape_dist.num_samples = 5000;
params.shape_dist.bin_edges = [0:5:150, inf];
params.shape_dist.si_bin_edges = linspace(0, 1, 20);
params.angle_edges = linspace(-1, 1, 10);

% parameters for transformations
params.icp.outlier_distance = 10;
params.transform_type = 'icp';
params.apply_known_mask = 1; % in aggragation, do we exploit known free space?
params.aggregating = true;

% do we plot certain steps in the matching and aggregating process?
params.plotting.plot_matches = 0;
params.plotting.num_matches = nan; % number of matches to plot in subplots
params.plotting.plot_transforms = 0;
params.plotting.plot_rotated_masks = 0; 

params.weights_threshold = 0.99;