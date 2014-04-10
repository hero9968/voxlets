function output = weights_predict_with_gt(model, depth, height, params, test_data_images, num)
% aim is to predict the output image using the GT to guide the selection of
% weights

transforms = propose_transforms(model, depth, params);
[~, ~, transformed] = aggregate_masks(transforms, height, depth, params);

% Here will try to optimise for the weights
% want to find the weights that minimise the sum of squared errors over the
% hidden part of the image
gt_img = single(test_data_images{num});
mask_stack = single(cell2mat(reshape({transformed.cropped_mask}, 1, 1, [])));
[~, other] = find_optimal_weights(depth, mask_stack, gt_img, params.optimisation_scale_factor);

% forming final output image
output = other.final_image;
padding = height - size(output, 1);
output = [output; zeros(padding, size(output, 2))];