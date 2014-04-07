function [weights_out, other] = find_optimal_weights(depth, mask_stack, gt_img)


% removing the pixels which are def. empty
gt_filled = fill_grid_from_depth(depth, size(mask_stack, 1), 0.5);
for ii = 1:size(mask_stack, 3)
    temp = mask_stack(:, :, ii);
    temp(gt_filled==0) = 0;
    mask_stack(:, :, ii) = temp;
end

% cropping gt image and the stacked grid
max_pred_depth = findfirst(squeeze(sum(sum(mask_stack, 2), 3))==0, 1, 1, 'first');
max_gt_depth = findfirst(squeeze(sum(gt_img, 2))==0, 1, 1, 'first');
max_depth = max(max_pred_depth, max_gt_depth);
gt_img = gt_img(1:max_depth, :);
mask_stack = mask_stack(1:max_depth, :, :);

% reshaping the inputs to be nice
mask_stack_trans = permute(mask_stack, [3, 1, 2]);
mask_stack_trans = double(mask_stack_trans(:, :));
gt_img = double(gt_img(:)');

% finally doing the optimisation
err_fun = @(x)((abs(gt_img - noisy_or(mask_stack_trans, 1, x))));

weights = ones(1, size(mask_stack, 3))/2;

options = optimoptions('lsqnonlin', 'TolFun', 1e-5, 'TolX', 1e-2);
[weights_out,Resnorm,Fval,flag, out] = lsqnonlin(err_fun, weights, ...
    zeros(size(weights)), ones(size(weights)), options);

% forming the output
other.EXITFLAG = flag;
other.lsqnonlin_out = out;
other.Resnorm = Resnorm;
other.Fval = Fval;
other.final_image = noisy_or(mask_stack, 3, weights_out);
other.height = max_depth;

