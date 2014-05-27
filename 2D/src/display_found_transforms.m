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
close all
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
num = 1000;
params.num_proposals = 200;
params.apply_known_mask = 0;
params.transform_type = 'pca';
params.icp.outlier_distance = 50;

depth = test_data(num).depth;
segments = test_data(num).segmented;
norms = test_data(num).normals;
raw_image = all_images{test_data(num).image_idx};
[~, gt_imgs] = rotate_and_raytrace_mask(raw_image, test_data(num).angle, 1);
gt_image = gt_imgs{1};

%profile on
transforms = propose_segmented_transforms(model, depth, norms, segments, params);
%profile off viewer

transforms2 = transforms(randperm(length(transforms)));

[out_img, out_img_cropped, transformed] = ...
    aggregate_masks(transforms2, size(gt_image, 1), depth, params);

%%
plot_transforms(transformed, out_img_cropped, gt_image);