function predictor = get_predictor(index, get_handle, params, paths, model)
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
        
        predictor.params = params;
        predictor.name = 'trained_gaussian';
        predictor.nicename = 'Trained Gaussian';
        predictor.shortname = 'Trained Gaussian';
        predictor.outpath = fullfile(paths.predictions, [predictor.name, '/']);
        
        if get_handle
            load(paths.gaussian_predict_model_path, 'model');
            predictor.handle = @(x, h, dummy1, dummy2)(gaussian_model_predict(model, x, h));
        end
        
    case 2
        
        params.scale_invariant = true;
        params.num_proposals = 300;
        params.transform_type = 'pca';

        predictor.params = params;
        predictor.name = 'gt_weighted_pca';
        predictor.nicename = 'Weighted aggregation of PCA, using GT img';
        predictor.shortname = 'Weighted using GT, PCA';
        predictor.outpath = fullfile(paths.predictions, [predictor.name, '/']);
        
        if get_handle
            %load(paths.structured_predict_si_model_path, 'model');
            predictor.handle = @(x, h, segments, gt_image)(weights_predict_with_gt_segmented(model, x, segments, h, params, gt_image));
        end
        
    case 3
        
        params.scale_invariant = true;
        params.num_proposals = 300;
        params.transform_type = 'icp';

        predictor.params = params;
        predictor.name = 'gt_weighted_icp';
        predictor.nicename = 'Weighted aggregation of ICP, using GT img';
        predictor.shortname = 'GT  Weighted, ICP';
        predictor.outpath = fullfile(paths.predictions, [predictor.name, '/']);
        
        if get_handle
            %load(paths.structured_predict_si_model_path, 'model');
            predictor.handle = @(x, h, segments, gt_image)(weights_predict_with_gt_segmented(model, x, segments, h, params, gt_image));
        end
        
    case 4

        params.scale_invariant = true;
        params.num_proposals = 10;
        params.transform_type = 'pca';
        
        predictor.params = params;
        predictor.name = 'gt_weighted_pca_seg';
        predictor.nicename = 'Weighted aggregation of SI, using PCA, and segmented';
        predictor.shortname = 'GT Weighted, PCA, segmented';
        predictor.outpath = fullfile(paths.predictions, [predictor.name, '/']);
        
        if get_handle
            load(paths.structured_predict_si_model_path, 'model');
            predictor.handle = @(x, h, dummy, gt_image)(weights_predict_with_gt_segmented(model, x, h, params, gt_image));
        end
            
    case 5
        
        params.scale_invariant = true;
        params.num_proposals = 10;
        params.transform_type = 'icp';
        
        predictor.params = params;
        predictor.name = 'gt_weighted_icp_seg';
        predictor.nicename = 'Weighted aggregation of SI, using ICP, and segmented';
        predictor.shortname = 'GT Weighted, ICP, segmented';
        predictor.outpath = fullfile(paths.predictions, [predictor.name, '/']);
        
        if get_handle
            load(paths.structured_predict_si_model_path, 'model');
            predictor.handle = @(x, h, dummy, gt_image)(weights_predict_with_gt_segmented(model, x, h, params, gt_image));
        end    
        
    otherwise
        
        error('No predictor with this index')
        
end

% ensuring folder paths exist
for ii = 1:length(predictor)
    if ~exist(predictor(ii).outpath, 'dir')
        mkdir(predictor(ii).outpath)
    end
end

