
% parameters for the predictors
predictor(1).name = 'pca_symmetry';
predictor(1).nicename = 'PCA symmetry';
predictor(1).handle = @(x)(pca_symmetry_predict(x, params));
predictor(1).outpath = fullfile(paths.predictions, 'pca_symmetry/');

load(paths.structured_predict_model_path, 'model');
params.scale_invariant = false;
predictor(2).name = 'structured_depth';
predictor(2).nicename = 'Structured depth';
predictor(2).handle = @(x)(test_fitting_model(model, x, params));
predictor(2).outpath = fullfile(paths.predictions, 'structured_depth/');

load(paths.gaussian_predict_model_path, 'model');
predictor(3).name = 'trained_gaussian';
predictor(3).nicename = 'Trained Gaussian';
predictor(3).handle = @(x)(gaussian_model_predict(model, x, params));
predictor(3).outpath = fullfile(paths.predictions, 'trained_gaussian/');

load(paths.structured_predict_si_model_path, 'model');
params.scale_invariant = true;
params.aggregating = true;
predictor(4).name = 'structured_depth_si';
predictor(4).nicename = 'Structured depth scale invariant';
predictor(4).handle = @(x)(test_fitting_model(model, x, params));
predictor(4).outpath = fullfile(paths.predictions, 'structured_depth_si/');

load(paths.structured_predict_si_model_path, 'model');
load(paths.test_data, 'test_data')
params.scale_invariant = true;
predictor(5).name = 'gt_weighted';
predictor(5).nicename = 'Weighted aggregation of SI, using GT img';
predictor(5).handle = @(x, y)(weights_predict_with_gt(model, x, params, test_data.images, y));
predictor(5).outpath = fullfile(paths.predictions, 'gt_weighted/');


