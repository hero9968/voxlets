function [weights_out, other] = find_optimal_weights(depth, mask_stack, gt_img)


% reshaping the inputs to be nice
mask_stack_trans = permute(mask_stack, [3, 1, 2]);
mask_stack_trans = double(mask_stack_trans(:, :));
gt_img_flat = double(gt_img(:)');

% removing the pixels which are def. empty and which no-one cares about
gt_filled = fill_grid_from_depth(depth, size(mask_stack, 1), 0.5);
gt_filled_flat = gt_filled(:)';

to_remove = gt_filled_flat == 0 | (gt_img_flat==0 & all(mask_stack_trans==0, 1));
%imagesc(reshape(to_remove, size(gt_img)));
gt_img_flat(to_remove) = [];
mask_stack_trans(:, to_remove) = [];

% finally doing the optimisation
err_fun = @(x)((abs(gt_img_flat - noisy_or(mask_stack_trans, 1, x))));

weights = ones(1, size(mask_stack, 3))/2;

options = optimoptions('lsqnonlin', 'TolFun', 1e-5, 'TolX', 1e-5);
[weights_out,Resnorm,Fval,flag, out] = lsqnonlin(err_fun, weights, ...
    zeros(size(weights)), ones(size(weights)), options);

% forming the output
other.EXITFLAG = flag;
other.lsqnonlin_out = out;
other.Resnorm = Resnorm;
other.Fval = Fval;
other.final_image = noisy_or(mask_stack, 3, weights_out);
other.final_image(gt_filled==0) = 0;
other.height = 100;
