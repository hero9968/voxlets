function [weights_out, other] = find_optimal_weights(depth, mask_stack, gt_img_in, scale_factor)
% optimisation to find the best set of weights.
% I am probably now replacing this with a function which does a much
% simpler weighting

assert(all(gt_img_in(:)<=1));
assert(all(gt_img_in(:)>=0));

% removing the pixels which are def. empty and which no-one cares about
gt_filled = fill_grid_from_depth(depth, size(mask_stack, 1), 0.5);

% finding a suitable scale factor ? cannot make it too small!
num_pixels = size(mask_stack, 1) * size(mask_stack, 2) - sum(~gt_filled(:));
min_pixels = size(mask_stack, 3);
min_area_scale_factor = min_pixels / num_pixels;
min_linear_scale_factor = sqrt(min_area_scale_factor) + 0.01;
scale_factor = max(scale_factor, min_linear_scale_factor);
disp(['scale_factor is ' num2str(scale_factor)])

% resizeing all images
gt_img = imresize(gt_img_in, scale_factor, 'bilinear');
mask_stack_resized = imresize(mask_stack, scale_factor, 'bilinear');
gt_filled_resized = imresize(gt_filled, scale_factor, 'bilinear');

% reshaping the inputs to be nice
mask_stack_trans = permute(mask_stack_resized, [3, 1, 2]);
mask_stack_trans = double(mask_stack_trans(:, :));
gt_img_flat = double(gt_img(:)');
gt_filled_flat = gt_filled_resized(:)';

% removing the pixels which are def. empty and which no-one cares about
to_remove = gt_filled_flat == 0 | (gt_img_flat==0 & all(mask_stack_trans==0, 1));
gt_img_flat(to_remove) = [];
mask_stack_trans(:, to_remove) = [];
size(mask_stack_trans)

% initialising the weights to ones, as want them to be big if possible
weights = zeros(1, size(mask_stack_resized, 3)) / 2;

% computing the true positives
size_prediction = sum(mask_stack_trans>0.5, 2);
N = size(mask_stack_trans, 1);
gt_img_repmat = repmat(gt_img_flat>0.5, N, 1);
size_true_positive = sum(mask_stack_trans>0.5 & gt_img_repmat, 2);
true_positive_fraction = size_true_positive ./ size_prediction;
true_positive_fraction(isnan(true_positive_fraction)) = 0;
false_positive_fraction = 1 - true_positive_fraction;

% finally doing the optimisation
gamma = 0.;
err_fun = @(x)(function_and_jacobian(x, mask_stack_trans, gt_img_flat, (true_positive_fraction(:))'));
options = optimoptions('lsqnonlin', 'TolFun', 1e-5, 'TolX', 1e-5, 'MaxIter', 1000, 'MaxFunEvals', 10000, 'Jacobian', 'off');%, 'DerivativeCheck', 'on');

min_weights = zeros(size(weights));
max_weights = ones(size(weights));

[weights_out, Resnorm, Fval, flag, out, ~, Jacout] = ...
    lsqnonlin(err_fun, weights, min_weights, max_weights, options);

% forming the output
other.EXITFLAG = flag;
other.lsqnonlin_out = out;
other.Resnorm = Resnorm;
other.Fval = Fval;
[other.final_image, T] = noisy_or(mask_stack, 3, weights_out);
[other.softmax, ~] = soft_max(mask_stack, 3, 10, weights_out);
other.final_image_added = sum(T, 3);
other.final_image(gt_filled==0) = 0;

[other.mean_image] = mean(T, 3);



function [E, J] = function_and_jacobian(x, mask_stack_trans, gt_img_flat, gamma)

%[~, M] = noisy_or(mask_stack_trans, 1, x);
alpha = 10;
%prediction = mean(M, 1);
[prediction, M] = soft_max(mask_stack_trans, 1, alpha, x);

% error is formed of per-pixel error, plus an error encouraging weights to be high
E = (prediction - gt_img_flat) .* (1 - gt_img_flat);
%E1 = prediction - gt_img_flat;
%E2 = 0.1*gamma .* (1-x);
%E = [E1, E2];

if nargout == 2

    % nb mask_stack_trans is B in my maths notation
    B = mask_stack_trans;
    T = repmat((1 - prediction), size(B, 1), 1);
    numerator = T.*B;
    denominator = 1 - M;
    
    J1 = (numerator ./ (denominator+0.0001))';
    J2 = -diag(gamma);
    J = [J1; J2];

end



