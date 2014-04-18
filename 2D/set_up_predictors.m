
% parameters for the predictors
%predictor(1).name = 'pca_symmetry';
%predictor(1).nicename = 'PCA symmetry';
%predictor(1).handle = @(x, h, dummy)(pca_symmetry_predict(x, h));
%predictor(1).outpath = fullfile(paths.predictions, 'pca_symmetry/');

load(paths.structured_predict_model_path, 'model');
params.scale_invariant = false;
params.num_proposals = 10;
predictor(2).name = 'structured_depth';
predictor(2).nicename = 'Structured depth';
predictor(2).shortname = 'Structured depth';
predictor(2).handle = @(x, h, dummy1, dummy2)(test_fitting_model(model, x, h, params));
predictor(2).outpath = fullfile(paths.predictions, 'structured_depth/');

load(paths.gaussian_predict_model_path, 'model');
predictor(3).name = 'trained_gaussian';
predictor(3).nicename = 'Trained Gaussian';
predictor(3).shortname = 'Trained Gaussian';
predictor(3).handle = @(x, h, dummy1, dummy2)(gaussian_model_predict(model, x, h));
predictor(3).outpath = fullfile(paths.predictions, 'trained_gaussian/');

load(paths.structured_predict_si_model_path, 'model');
params.scale_invariant = true;
params.num_proposals = 10;
predictor(4).name = 'structured_depth_si';
predictor(4).nicename = 'Structured depth scale invariant';
predictor(4).shortname = 'Structured predict (SI)';
predictor(4).handle = @(x, h, dummy1, dummy2)(test_fitting_model(model, x, h, params));
predictor(4).outpath = fullfile(paths.predictions, 'structured_depth_si/');

load(paths.structured_predict_si_model_path, 'model');
load(paths.test_data, 'test_data')
params.scale_invariant = true;
params.num_proposals = 10;
params.optimisation_scale_factor = 1; % in the gt optimisation, the 
predictor(1).name = 'gt_weighted';
predictor(1).nicename = 'Weighted aggregation of SI, using GT img';
predictor(1).shortname = 'Weighted using GT';
predictor(1).handle = @(x, h, dummy, y)(weights_predict_with_gt(model, x, h, params, test_data.images, y));
predictor(1).outpath = fullfile(paths.predictions, 'gt_weighted/');


load(paths.structured_predict_si_model_path, 'model');
load(paths.test_data, 'test_data')
params.scale_invariant = true;
params.num_proposals = 30;
params.optimisation_scale_factor = 0.1; % in the gt optimisation, the 
predictor(5).name = 'gt_weighted_scaled';
predictor(5).nicename = 'Weighted aggregation of SI, using GT img - scaled';
predictor(5).shortname = 'Weighted using GT (ICP)';
predictor(5).handle = @(x, h, dummy, y)(weights_predict_with_gt(model, x, h, params, test_data.images, y));
predictor(5).outpath = fullfile(paths.predictions, 'gt_weighted_scaled/');

load(paths.structured_predict_si_model_path, 'model');
load(paths.test_data, 'test_data')
params.scale_invariant = true;
params.num_proposals = 30;
params.transform_type = 'pca';
params.optimisation_scale_factor = 0.1; % in the gt optimisation, the 
predictor(6).name = 'gt_weighted_scaled_pca';
predictor(6).nicename = 'Weighted aggregation of SI, using GT img - scaled';
predictor(6).shortname = 'Weighted using GT (PCA)';
predictor(6).handle = @(x, h, dummy, y)(weights_predict_with_gt(model, x, h, params, test_data.images, y));
predictor(6).outpath = fullfile(paths.predictions, 'gt_weighted_scaled_pca/');


load(paths.structured_predict_si_model_path, 'model');
load(paths.test_data, 'test_data')
params.scale_invariant = true;
params.num_proposals = 50;
params.transform_type = 'pca';
params.optimisation_scale_factor = 0.1; % in the gt optimisation, the 
predictor(7).name = 'gt_weighted_scaled_pca_seg';
predictor(7).nicename = 'Weighted aggregation of SI, using PCA, and segmented';
predictor(7).shortname = 'Weighted, PCA, segmented';
predictor(7).handle = @(x, h, s, y)(weights_predict_with_gt_segmented(model, x, s, h, params, test_data.images, y));
predictor(7).outpath = fullfile(paths.predictions, 'gt_weighted_scaled_pca_segmented/');

load(paths.structured_predict_si_model_path, 'model');
load(paths.test_data, 'test_data')
params.scale_invariant = true;
params.num_proposals = 50;
params.transform_type = 'icp';
params.optimisation_scale_factor = 0.1; % in the gt optimisation, the 
predictor(8).name = 'gt_weighted_scaled_icp_seg';
predictor(8).nicename = 'Weighted aggregation of SI, using ICP, and segmented';
predictor(8).shortname = 'Weighted, ICP, segmented';
predictor(8).handle = @(x, h, s, y)(weights_predict_with_gt_segmented(model, x, s, h, params, test_data.images, y));
predictor(8).outpath = fullfile(paths.predictions, 'gt_weighted_scaled_icp_segmented/');


for ii = 1:length(predictor)
    if ~exist(predictor(ii).outpath, 'dir')
        mkdir(predictor(ii).outpath)
    end
end

