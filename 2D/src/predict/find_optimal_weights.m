function [weights_out, other] = find_optimal_weights(depth, mask_stack, gt_img, scale_factor)

assert(all(gt_img(:)<=1));
assert(all(gt_img(:)>=0));

% removing the pixels which are def. empty and which no-one cares about
gt_filled = fill_grid_from_depth(depth, size(mask_stack, 1), 0.5);

% finding a suitable scale factor ? cannot make it too small!
num_pixels = size(mask_stack, 1) * size(mask_stack, 2);
min_pixels = size(mask_stack, 3);
min_area_scale_factor = min_pixels / num_pixels;
min_linear_scale_factor = sqrt(min_area_scale_factor) + 0.01;
scale_factor = max(scale_factor, min_linear_scale_factor);

% resizeing all images
gt_img = imresize(gt_img, scale_factor, 'bilinear');
mask_stack_resized = imresize(mask_stack, scale_factor, 'bilinear');
gt_filled_resized = imresize(gt_filled, scale_factor, 'bilinear');

% reshaping the inputs to be nice
mask_stack_trans = permute(mask_stack_resized, [3, 1, 2]);
mask_stack_trans = double(mask_stack_trans(:, :));
gt_img_flat = double(gt_img(:)');
gt_filled_flat = gt_filled_resized(:)';

to_remove = gt_filled_flat == 0 | (gt_img_flat==0 & all(mask_stack_trans==0, 1));
%imagesc(reshape(to_remove, size(gt_img)));
gt_img_flat(to_remove) = [];
mask_stack_trans(:, to_remove) = [];

% initialising the weights
weights = ones(1, size(mask_stack_resized, 3))/2;
assert(size(mask_stack_trans, 1)==length(weights));
for ii = 1:size(mask_stack_trans, 1)
    size_prediction = sum(mask_stack_trans(ii, :));
    size_true_positive = sum(mask_stack_trans(ii, gt_img_flat>0.5));
    if size_true_positive == 0
        weights(ii) = 0;
    else
        weights(ii) = size_true_positive / size_prediction;
    end
    assert(weights(ii) >= 0 && weights(ii) <= 1);
end

% finally doing the optimisation
err_fun = @(x)((abs(gt_img_flat - noisy_or(mask_stack_trans, 1, x))));
options = optimoptions('lsqnonlin', 'TolFun', 1e-5, 'TolX', 1e-5);

min_weights = zeros(size(weights));
max_weights = ones(size(weights));
[weights_out, Resnorm, Fval, flag, out] = ...
    lsqnonlin(err_fun, weights, min_weights, max_weights, options);

% forming the output
other.EXITFLAG = flag;
other.lsqnonlin_out = out;
other.Resnorm = Resnorm;
other.Fval = Fval;
other.final_image = noisy_or(mask_stack, 3, weights_out);
other.final_image(gt_filled==0) = 0;
other.height = 100;
