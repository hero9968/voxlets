paths.data = '~/projects/shape_sharing/data/';
paths.mpeg = [paths.data, '2D_shapes/MPEG7_CE-Shape-1_Part_B/'];
paths.subset = [paths.data, '2D_shapes/MPEG7_subset/'];

paths.subset_files = [paths.subset, 'filelist.mat'];

paths.rotated = [paths.data, '2D_shapes/rotated/'];
paths.rotated_filename = '%02d_%02d_mask.gif';
paths.rotated_savename = [paths.rotated, paths.rotated_filename];

paths.raytraced = [paths.data, '2D_shapes/raytraced/'];
paths.raytraced_savename = [paths.raytraced, '%02d_%02d_mask.png'];

% train/test split
paths.split_path = [paths.data, '2D_shapes/split.mat'];

paths.predictions = [paths.data, '2D_shapes/predict/'];

% params for setting up file lists
params.number_subclasses = 1; % how many subclasses from each shape to use

% angles to rotate masks
params.n_angles = 16;
temp_angles = linspace(0, 360, params.n_angles+1);
params.angles = temp_angles(1:end-1);
clear temp_angles

% size of output image
params.im_height = 250;
%params.im_width = 50;
params.scale = 0.2;

% some hand-defined prediction models
params.gauss_model.mu = 0;
params.gauss_model.sigma = 10;

% feature computation params
params.shape_dist.num_samples = 2000;
params.shape_dist.bin_edges = [0:5:150, inf];


% parameters for the predictors
predictor(1).name = 'per_ray_gaussian';
predictor(1).handle = @(x)(per_ray_gaussian_prediction(x, params.gauss_model, params));
predictor(1).outpath = fullfile(paths.predictions, 'per_ray_gaussian/');

predictor(2).name = 'pca_symmetry';
predictor(2).handle = @(x)(pca_symmetry_predict(x, params));
predictor(2).outpath = fullfile(paths.predictions, 'pca_symmetry/');

