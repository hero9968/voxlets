function output = weights_predict_with_gt(model, depth, height, params, gt_image)
% aim is to predict the output image using the GT to guide the selection of
% weights

%{
load(paths.structured_predict_si_model_path, 'model')
params.scale_invariant = true;
params.num_proposals = 30;
params.optimisation_scale_factor = 0.1; % in the gt optimisation, the 
params.transform_type = 'pca'
%}

%%
transforms = propose_transforms(model, depth, params);
[~, ~, transformed] = aggregate_masks(transforms, height, depth, params);

%%
if 0
    for ii = 1:length(transformed)
        subplot(8, 8, ii);
        imagesc(transformed(ii).cropped_mask);
        axis image
    end
end

%% Here will try to optimise for the weights
% want to find the weights that minimise the sum of squared errors over the
% hidden part of the image
mask_stack = single(cell2mat(reshape({transformed.cropped_mask}, 1, 1, [])));
%[W, other] = find_optimal_weights(depth, mask_stack, gt_image, params.optimisation_scale_factor);
[W, other] = find_best_weights_simple(depth, mask_stack, gt_image, params.weights_threshold);

%%

%final_image = noisy_or(mask_stack, 3, W);
if 0
    subplot(131)
    imagesc(gt_image);
    axis image
    subplot(132)
    imagesc(other.final_image)
    axis image
    subplot(133)
    colorbar
end

%% forming final output image
output = other.simple_image;
padding = height - size(output, 1);
output = [output; zeros(padding, size(output, 2))];

