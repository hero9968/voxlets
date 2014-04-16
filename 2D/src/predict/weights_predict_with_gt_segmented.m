function output = weights_predict_with_gt_segmented(model, depth, segments, height, params, test_data_images, num)
% aim is to predict the output image using the GT to guide the selection of
% weights

assert(isscalar(height));
assert(size(segments, 2) == length(depth));
assert(isvector(depth))

transforms = propose_segmented_transforms(model, depth, segments, params);
[~, ~, transformed] = aggregate_masks(transforms, height, depth, params);

% Here will try to optimise for the weights
% want to find the weights that minimise the sum of squared errors over the
% hidden part of the image
gt_img = single(test_data_images{num});
mask_stack = single(cell2mat(reshape({transformed.extended_mask}, 1, 1, [])));
[~, other] = find_optimal_weights(depth, mask_stack, gt_img, params.optimisation_scale_factor);

% forming final output image
output = other.final_image;
padding = height - size(output, 1);
output = [output; zeros(padding, size(output, 2))];