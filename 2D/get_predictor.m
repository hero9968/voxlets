function predictor = get_predictor(index, get_handle, params, paths)
% returns a structure of information about a prediction algorithm
% ... such as its name, parameters structure, path where the results
% are saved etc. 
% If get_handle is true, a handle to enable the prediction to be made is
% also returned. This is not returned by default as it typically involves
% loading in a large model.


% parameters for the predictors
%predictor(1).name = 'pca_symmetry';
%predictor(1).nicename = 'PCA symmetry';
%predictor(1).handle = @(x, h, dummy)(pca_symmetry_predict(x, h));
%predictor(1).outpath = fullfile(paths.predictions, 'pca_symmetry/');

if ~isscalar(index)
    for ii = 1:length(index)
        predictor(ii) = get_predictor(index(ii), get_handle, params, paths);
    end
    return
end


switch index
    
    case 1
        
        params.scale_invariant = true;
        params.num_proposals = 10;
        params.optimisation_scale_factor = 1; % in the gt optimisation, the 
        
        predictor.params = params;
        predictor.name = 'gt_weighted';
        predictor.nicename = 'Weighted aggregation of SI, using GT img';
        predictor.shortname = 'Weighted using GT';
        predictor.outpath = fullfile(paths.predictions, 'gt_weighted/');
        
        if get_handle
            load(paths.structured_predict_si_model_path, 'model');
            load(paths.test_data, 'test_data')
            predictor.handle = @(x, h, dummy, y)(weights_predict_with_gt(model, x, h, params, {test_data.image}, y));
        end
        
    case 2
        
        params.scale_invariant = false;
        params.num_proposals = 10;
        
        predictor.params = params;
        predictor.name = 'structured_depth';
        predictor.nicename = 'Structured depth';
        predictor.shortname = 'Structured depth';
        predictor.outpath = fullfile(paths.predictions, 'structured_depth/');
        
        if get_handle
            load(paths.structured_predict_model_path, 'model');
            predictor.handle = @(x, h, dummy1, dummy2)(test_fitting_model(model, x, h, params));
        end

    case 3
        
        predictor.params = params;
        predictor.name = 'trained_gaussian';
        predictor.nicename = 'Trained Gaussian';
        predictor.shortname = 'Trained Gaussian';
        predictor.outpath = fullfile(paths.predictions, 'trained_gaussian/');
        
        if get_handle
            load(paths.gaussian_predict_model_path, 'model');
            predictor.handle = @(x, h, dummy1, dummy2)(gaussian_model_predict(model, x, h));
        end
        
    case 4
        
        params.scale_invariant = true;
        params.num_proposals = 10;
        
        predictor.params = params;
        predictor.name = 'structured_depth_si';
        predictor.nicename = 'Structured depth scale invariant';
        predictor.shortname = 'Structured predict (SI)';
        predictor.outpath = fullfile(paths.predictions, 'structured_depth_si/');
        
        if get_handle
            load(paths.structured_predict_si_model_path, 'model');
            predictor.handle = @(x, h, dummy1, dummy2)(test_fitting_model(model, x, h, params));
        end
        
    case 5        
        
        params.scale_invariant = true;
        params.num_proposals = 30;
        params.optimisation_scale_factor = 0.1; % in the gt optimisation, the 
        
        predictor.params = params;
        predictor.name = 'gt_weighted_scaled';
        predictor.nicename = 'Weighted aggregation of SI, using GT img - scaled';
        predictor.shortname = 'Weighted using GT (ICP)';
        predictor.outpath = fullfile(paths.predictions, 'gt_weighted_scaled/');
        
        if get_handle
            load(paths.structured_predict_si_model_path, 'model');
            load(paths.test_data, 'test_data');
            predictor.handle = @(x, h, dummy, y)(weights_predict_with_gt(model, x, h, params, {test_data.image}, y)); 
        end
        
    case 6

        params.scale_invariant = true;
        params.num_proposals = 30;
        params.transform_type = 'pca';
        params.optimisation_scale_factor = 0.1; % in the gt optimisation, the 
        
        predictor.params = params;
        predictor.name = 'gt_weighted_scaled_pca';
        predictor.nicename = 'Weighted aggregation of SI, using GT img - scaled';
        predictor.shortname = 'Weighted using GT (PCA)';
        predictor.outpath = fullfile(paths.predictions, 'gt_weighted_scaled_pca/');
        
        if get_handle
            load(paths.structured_predict_si_model_path, 'model');
            load(paths.test_data, 'test_data')
            predictor.handle = @(x, h, dummy, y)(weights_predict_with_gt(model, x, h, params, {test_data.image}, y));
        end
        
    case 7

        params.scale_invariant = true;
        params.num_proposals = 150;
        params.transform_type = 'pca';
        params.optimisation_scale_factor = 0.1; % in the gt optimisation, the 
        
        predictor.params = params;
        predictor.name = 'gt_weighted_scaled_pca_seg';
        predictor.nicename = 'Weighted aggregation of SI, using PCA, and segmented';
        predictor.shortname = 'Weighted, PCA, segmented';
        predictor.outpath = fullfile(paths.predictions, 'gt_weighted_scaled_pca_segmented/');
        
        if get_handle
            load(paths.structured_predict_si_model_path, 'model');
            load(paths.test_data, 'test_data')
            predictor.handle = @(x, h, s, y)(weights_predict_with_gt_segmented(model, x, s, h, params, {test_data.image}, y));
        end
            
    case 8

        params.scale_invariant = true;
        params.num_proposals = 150;
        params.transform_type = 'icp';
        params.optimisation_scale_factor = 0.1; % in the gt optimisation, the 
        
        predictor.params = params;
        predictor.name = 'gt_weighted_scaled_icp_seg';
        predictor.nicename = 'Weighted aggregation of SI, using ICP, and segmented';
        predictor.shortname = 'Weighted, ICP, segmented';
        predictor.outpath = fullfile(paths.predictions, 'gt_weighted_scaled_icp_segmented/');
        
        if get_handle
            load(paths.structured_predict_si_model_path, 'model');
            load(paths.test_data, 'test_data')
            predictor.handle = @(x, h, s, y)(weights_predict_with_gt_segmented(model, x, s, h, params, {test_data.image}, y));
        end
        
end

% ensuring folder paths exist
for ii = 1:length(predictor)
    if ~exist(predictor(ii).outpath, 'dir')
        mkdir(predictor(ii).outpath)
    end
end

