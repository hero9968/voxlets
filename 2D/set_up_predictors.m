
% parameters for the predictors
predictor(1).name = 'per_ray_gaussian';
predictor(1).nicename = 'Per ray Gaussian';
predictor(1).handle = @(x)(per_ray_gaussian_prediction(x, params.gauss_model, params));
predictor(1).outpath = fullfile(paths.predictions, 'per_ray_gaussian/');

predictor(2).name = 'pca_symmetry';
predictor(2).nicename = 'PCA symmetry';
predictor(2).handle = @(x)(pca_symmetry_predict(x, params));
predictor(2).outpath = fullfile(paths.predictions, 'pca_symmetry/');

load(paths.structured_predict_model_path, 'model');
params.scale_invariant = false;
predictor(3).name = 'structured_depth';
predictor(3).nicename = 'Structured depth';
predictor(3).handle = @(x)(test_fitting_model(model, x, params));
predictor(3).outpath = fullfile(paths.predictions, 'structured_depth/');

load(paths.gaussian_predict_model_path, 'model');
predictor(4).name = 'trained_gaussian';
predictor(4).nicename = 'Trained Gaussian';
predictor(4).handle = @(x)(gaussian_model_predict(model, x, params));
predictor(4).outpath = fullfile(paths.predictions, 'trained_gaussian/');

load(paths.structured_predict_si_model_path, 'model');
params.scale_invariant = true;
predictor(5).name = 'structured_depth_si';
predictor(5).nicename = 'Structured depth scale invariant';
predictor(5).handle = @(x)(gaussian_model_predict(model, x, params));
predictor(5).outpath = fullfile(paths.predictions, 'structured_depth_si/');