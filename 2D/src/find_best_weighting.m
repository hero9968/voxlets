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
gt_image = rotate_image_nicely(raw_image, test_data(num).angle);

%%
angs = linspace(0, 360, 100);
for ii = 1:length(angs)
    gt_image = rotate_image_nicely(raw_image, angs(ii));
    subplot(10, 10, ii);
    imagesc(gt_image);
    axis image
end
%%
params.transform_type = 'icp';

% propose transforms and aggregate
params.im_min_height = size(gt_image, 1);
%transforms = propose_transforms(model, depth, params);
transforms = propose_segmented_transforms(model, depth, segments, params);
[out_img, out_img_cropped, transformed] = ...
        aggregate_masks(transforms, params.im_min_height, depth, params);

plot_transforms(transformed, out_img_cropped, gt_image);

%% now am going to try to get proposals from each of the segments
num = 5200;
params.num_proposals = 200;
params.apply_known_mask = 0;
params.transform_type = 'pca';
params.icp.outlier_distance = 50;

depth = test_data(num).depth;
segments = test_data(num).segmented;
raw_image = all_images{test_data(num).image_idx};
[~, gt_imgs] = rotate_and_raytrace_mask(raw_image, test_data(num).angle, 1);
gt_image = gt_imgs{1};

profile on
transforms = propose_segmented_transforms(model, depth, segments, params);
profile off viewer


transforms2 = transforms(randperm(length(transforms)));

[out_img, out_img_cropped, transformed] = ...
    aggregate_masks(transforms2, size(gt_image, 1), depth, params);

%%
plot_transforms(transformed, out_img_cropped, gt_image);

%% Here will try to optimise for the weights
% want to find the weights that minimise the sum of squared errors over the
% hidden part of the image
tic
mask_stack = single(cell2mat(reshape({transformed.cropped_mask}, 1, 1, [])));
opts.least_squares = 0;
%profile on
[weights, other] = find_optimal_weights(depth, mask_stack, im2double(gt_image), 0.1);
%profile off viewer
sum(weights > 0.8)
toc

%%
subplot(231)
imagesc(gt_image)
axis image

subplot(232)
imagesc(other.softmax)
axis image
set(gca, 'clim', [0, 1])
colormap(gray)

subplot(233)
plot(weights)
ylim([0, 1])

subplot(234)
hist(weights, 50)
xlim([0, 1])
%%

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
%profile on
[weights, other] = find_best_weights_simple(depth, mask_stack, im2double(gt_image), 0.8);
%profile off viewer

subplot(231)
imagesc(gt_image)
axis image

subplot(232)
imagesc(other.simple_image)
axis image
colormap(gray)

w2 = other.size_true_positive ./ other.size_prediction;
w2(isnan(w2)) = 0;
weighted_basis = mask_stack;
N = size(mask_stack, 3);
for ii = 1:N
    weighted_basis(:, :, ii, :) = weighted_basis(:, :, ii, :) * w2(ii); 
end


%%
clf
v_w = 0.5;
t_w = 0.9;
b = (v_w - t_w^2) / (t_w - t_w^2);
a = 1 - b;
t = 0:0.01:1;
f = @(x)(max(a * x.^2 + b * x, 0));
%f = @(x)(sin(pi*x/2) + cos(pi*x/2));
plot(t, f(t))
axis image

%%
clf
a = 1.1;
f = @(x)(max((x-a)./(1-a), 0));
%f = @(x)(sin(pi*x/2) + cos(pi*x/2));
plot(t, f(t))
axis image



%%
gt = im2double(gt_image);
dtrans = bwdist(edge(gt));
dtrans(gt==1) = 0;
subplot(233);
imagesc(dtrans)
axis image
colorbar
%%
alpha = 1000;

final_image = soft_max(mask_stack, 3, alpha, w2);

subplot(233);
imagesc2(final_image);
axis image
set(gca, 'clim', [0, 1])

subplot(234);
imagesc2(max(weighted_basis, [], 3));
set(gca, 'clim', [0, 1])


%%
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

%%
A = [0.1, 0.2, 0.3, 0.1]';
A = [A, A]
[T] = soft_max(A, 1, 10)

