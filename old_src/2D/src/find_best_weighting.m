% the aim of this script is to find a good weighting for the basis
% functions, given that we know the ground truth mask.

cd ~/projects/shape_sharing/2D/src
clear
run ../define_params
addpath predict
addpath utils
addpath transformations/
addpath(genpath('external'))
addpath ../../common/

%% loading in model and test data
load(paths.test_data, 'test_data')
load(paths.all_images, 'all_images')
load(paths.structured_predict_si_model_path, 'model');

%% now am going to try to get proposals from each of the segments
num = 200;
this_img = test_data(num);
this_img.raw_image = all_images{this_img.image_idx};
this_img.gt_image = rotate_image_nicely(this_img.raw_image, this_img.angle, length(this_img.depth));

params.num_proposals = 200;
params.apply_known_mask = 0;
params.transform_type = 'icp';
params.icp.outlier_distance = 50;

transforms = propose_segmented_transforms(model, this_img.depth, this_img.normals, this_img.segmented, params);
[out_img, out_img_cropped, transformed] = ...
    aggregate_masks(transforms, size(this_img.gt_image, 1), this_img.depth, params);
mask_stack = single(cell2mat(reshape({transformed.cropped_mask}, 1, 1, [])));

%% Global weight optimisation
% want to find the weights that minimise the sum of squared errors over the
% hidden part of the image

opts.least_squares = 0;
%profile on
[weights, other] = find_optimal_weights(this_img.depth, mask_stack, im2double(this_img.gt_image), 0.1);
%profile off viewer
sum(weights > 0.8)

%% Plot the gt vs the one from the optimiser
subplot(221)
imagesc2(this_img.gt_image)

subplot(222)
imagesc2(other.softmax)
set(gca, 'clim', [0, 1])

subplot(223)
plot(weights)
ylim([0, 1])

subplot(224)
hist(weights, 50)
xlim([0, 1])

%% Individual weight optimisation
%profile on
[weights, other] = find_best_weights_simple(this_img.depth, mask_stack, im2double(this_img.gt_image), 0.5);
%profile off viewer

%% Plotting these simple weights
subplot(231)
imagesc(this_img.gt_image)
axis image

subplot(232)
imagesc(other.summed_image)
axis image
colormap(gray)
set(gca, 'clim', [0, 1])

alpha = 1000;
w2 = other.size_true_positive ./ other.size_prediction;
w2(isnan(w2)) = 0;
soft_max_image = soft_max(mask_stack, 3, alpha, weights);

subplot(233);
imagesc2(soft_max_image);
axis image
set(gca, 'clim', [0, 1])

weighted_basis = mask_stack;
N = size(mask_stack, 3);
for ii = 1:N
    weighted_basis(:, :, ii, :) = weighted_basis(:, :, ii, :) * w2(ii); 
end

subplot(234);
imagesc2(max(weighted_basis, [], 3));
set(gca, 'clim', [0, 1])

subplot(235);
hist(weights, 50)
