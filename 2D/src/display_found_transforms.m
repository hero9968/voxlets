% propose transforms for an image and plot them nicely

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

%% loading in the data
num = 100;
this_img = test_data(num);
this_img.raw_image = all_images{this_img.image_idx};
this_img.gt_image = rotate_image_nicely(this_img.raw_image, this_img.angle);

%% diplaying the raw img rotated to different angles
close all
angs = linspace(0, 360, 100);
for ii = 1:length(angs)
    temp_rotated_img = rotate_image_nicely(this_img.raw_image, angs(ii));
    subplot(10, 10, ii);
    imagesc(temp_rotated_img);
    axis image
end

%% setting parameters for the transformation proposals
params.num_proposals = 200;
params.apply_known_mask = 0;
params.transform_type = 'icp';
params.icp.outlier_distance = 50;

%% propose transforms and aggregate
%profile on
%transforms = propose_transforms(model, depth, params);
transforms = ...
    propose_segmented_transforms(model, this_img.depth, this_img.normals, this_img.segmented, params);
%profile off viewer

transforms2 = transforms(randperm(length(transforms)));
[out_img, out_img_cropped, transformed] = ...
    aggregate_masks(transforms2, size(this_img.gt_image, 1), this_img.depth, params);

%%
plot_transforms(transformed, out_img_cropped, this_img.gt_image);

