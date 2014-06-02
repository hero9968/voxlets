% a script to train a model from the training data...

cd ~/projects/shape_sharing/2D
clear
define_params
addpath(genpath('.'));
addpath ../common/

%% loading in all depths and shapes from disk...
load(paths.all_images, 'all_images')
load(paths.train_data, 'train_data')

%% now compute the model
%train_data_subset = train_data;
%train_data_subset = train_data(randperm(length(train_data), 2000));
model = train_fitting_model(all_images, train_data, params);

%% showing all the shape distributions as an image
all_dists = cell2mat({model.training_data.shape_dist}');
close all
imagesc(all_dists)

%% save the model
save(paths.structured_predict_si_model_path, 'model');

%%
load(paths.test_data, 'test_data')

%% making a single prediction and visualising
%profile on

clf
num = 2401;
params.aggregating = 1;
params.num_proposals= 5;
params.plotting.plot_transforms = 1;
params.plotting.plot_matches = 0;
params.icp.outlier_distance = 10; 

for ii = 1:3
    
    subplot(2, 3, ii); 
    
    % extracting and transforming the GT image
    this_image_idx = test_data(num).image_idx;
    this_transform = test_data(num).transform.tdata.T';
    this_image = all_images{this_image_idx};
    this_test_image = transform_image(this_image, this_transform);
    
    % combinging with the depth for a nice display image
    combine_mask_and_depth(this_test_image, test_data(num).depth)
    width = length(test_data(num).depth);
    
end

height = size(this_test_image, 1);
params.transform_type = 'pca';
[stacked_image_pca, transforms] = test_fitting_model(model, test_data(num).depth, test_data(num).normals, height, params);
params.transform_type = 'icp';
[stacked_image_icp, transforms] = test_fitting_model(model, test_data(num).depth, test_data(num).normals, height, params);

% plot final stacked image
subplot(2, 3, 5);
imagesc(stacked_image_pca); axis image
subplot(2, 3, 6);
imagesc(stacked_image_icp); axis image

% formatting axes now
for ii = 1:6
    subplot(2, 3, ii); 
    set(gca, 'xlim', round([-width/4, 1.25*width]));
    set(gca, 'ylim',round([-width/4, 1.25*width]));
end
colormap(jet)
%profile off viewer

%% showing the closest matching features to the input image
close all
%load(paths.structured_predict_si_model_path, 'model');

params.aggregating = 1;
params.plotting.plot_matches = 1;
params.plotting.num_matches = 15;
params.plotting.plot_transforms = 1;
params.num_proposals= 20;

% finding the nearest neighbours
this_image_idx = test_data(num).image_idx;
this_transform = test_data(num).transform.tdata.T';
this_image = all_images{this_image_idx};
this_test_image = transform_image(this_image, this_transform);
height = size(this_test_image, 1);

S = test_fitting_model(model, test_data(num).depth, test_data(num).normals, height, params);
%%
% showing image of the input data and ground truth
subplot(3, 4, 3*4);
%combine_mask_and_depth(test_data(num).image, test_data(num).raytraced)
title('Ground truth')


