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

%% 
clf
num = 12;
params.num_proposals = 12;
params.apply_known_mask = 0;

% get the test data and the gt image
depth = test_data(num).depth;
segments = test_data(num).segmented;
raw_image = all_images{test_data(num).image_idx};
[~, gt_imgs] = rotate_and_raytrace_mask(raw_image, test_data(num).angle, 1);
gt_image = gt_imgs{1};
params.transform_type = 'pca';

% propose transforms and aggregate
transforms = propose_transforms(model, depth, params);
[out_img, out_img_cropped, transformed] = ...
        aggregate_masks(transforms, params.im_min_height, depth, params);

plot_transforms(transformed, out_img_cropped, gt_image);

%% now am going to try to get proposals from each of the segments
num = 200;
params.num_proposals = 200;
params.apply_known_mask = 0;
params.transform_type = 'icp';
params.icp.outlier_distance = 50;

depth = test_data(num).depth;
segments = test_data(num).segmented;
raw_image = all_images{test_data(num).image_idx};
[~, gt_imgs] = rotate_and_raytrace_mask(raw_image, test_data(num).angle, 1);
gt_image = gt_imgs{1};

transforms = propose_segmented_transforms(model, depth, segments, params);

transforms2 = transforms(randperm(length(transforms)));

[out_img, out_img_cropped, transformed] = ...
    aggregate_masks(transforms2, size(gt_image, 1), depth, params);

%%
plot_transforms(transformed, out_img_cropped, gt_image);

%% Here will try to optimise for the weights
% want to find the weights that minimise the sum of squared errors over the
% hidden part of the image
tic
mask_stack = single(cell2mat(reshape({transformed.extended_mask}, 1, 1, [])));
opts.least_squares = 0;
%profile on
[weights, other] = find_optimal_weights(depth, mask_stack, im2double(gt_image), 0.1);
%profile off viewer
toc

%%
subplot(231)
imagesc(gt_image)
axis image

subplot(232)
imagesc(other.final_image)
axis image
%set(gca, 'clim', [0, 1])
colormap(gray)


subplot(233)
imagesc(other.final_image_added)
axis image
%set(gca, 'clim', [0, 1])
colormap(gray)

subplot(235)
imagesc(other.simple_image)
axis image
%set(gca, 'clim', [0, 1])
colormap(gray)

subplot(236)
imagesc(other.simple_image_added)
axis image
%set(gca, 'clim', [0, 1])
colormap(gray)


%%
profile on
[weights, other] = find_best_weights_simple(depth, mask_stack, im2double(gt_image), 0.99);
profile off viewer

subplot(231)
imagesc(gt_image)
axis image

subplot(232)
imagesc(other.simple_image)
axis image
colormap(gray)

subplot(233)
imagesc(other.simple_image2)
axis image
colormap(gray)



%%
clf
[X, Y] = find(edge(test_data.images{num}));
x_loc = 0;%transformed(1).padding;
y_loc = 0;%transformed(1).padding;
xy = apply_transformation_2d([X'; Y'], [1, 0, y_loc; 0, 1, x_loc; 0 0 1]);

imagesc(out_img_cropped);
hold on
plot(xy(2, :), xy(1, :), 'r+')
hold off
colormap(flipgray)
%set(gca, 'clim', [0, 1])
axis image





%%
params.aggregating = 1;
num = 144
for ii = 1:3
    subplot(1, 4,ii); 
    combine_mask_and_depth(test_data.images{num}, test_data.depths{num})
    width = length(test_data.depths{num})
    set(gca, 'xlim', round([-width/2, 2.5*width]));
    set(gca, 'ylim',round([-width/2, 2.5*width]));
end

stacked_image = test_fitting_model(model, test_data.depths{num}, params);
subplot(1, 4, 4);
imagesc(stacked_image); axis image
