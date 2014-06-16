% propose transforms for an image and plot them nicely

cd ~/projects/shape_sharing/2D/src
clear
run ../define_params
addpath predict
addpath utils
addpath transformations/
addpath(genpath('external'))
addpath(genpath('../../common/'))

%% loading in model and test data
load(paths.test_data, 'test_data')
load(paths.all_images, 'all_images')
load(paths.structured_predict_si_model_path, 'model');

%% loading in the data
num = 1300;
this_img = test_data(num);
this_img.raw_image = all_images{this_img.image_idx};
this_img.gt_image = rotate_image_nicely(this_img.raw_image, this_img.angle);

%% diplaying the raw img rotated to different angles
if 0
    close all
    angs = linspace(0, 360, 100);
    for ii = 1:length(angs)
        temp_rotated_img = rotate_image_nicely(this_img.raw_image, angs(ii));
        subplot(10, 10, ii);
        imagesc(temp_rotated_img);
        axis image
    end
end

%% setting parameters for the transformation proposals
params.num_proposals = 100;
params.apply_known_mask = 0;
params.transform_type = 'icp';
params.icp.outlier_distance = 50;

%% propose transforms and aggregate
profile on
tic
%transforms = propose_transforms(model, depth, params);
transforms = ...
    propose_segmented_transforms(model, this_img.depth, this_img.normals, this_img.segmented, params);

%%
%transforms2 = transforms(randperm(length(transforms)));
[out_img, out_img_cropped, transformed] = ...
    aggregate_masks(transforms, size(this_img.gt_image, 1), this_img.depth, params);
%%
%
gt_depth = raytrace_2d(this_img.gt_image);

for ii = 1:length(transformed)
    
    % this prediction position
    X = transformed(ii).transformed_depth(1, :) - transformed(ii).padding;
    Y = transformed(ii).transformed_depth(2, :) - transformed(ii).padding;
    
    % doing the average neighbour distance
    XY_model = xy_from_depth(gt_depth);
    XY_data = [X; Y];
    transformed(ii).dist_to_gt = average_neighbour_distance(XY_model, XY_data);
    
end
toc
profile off viewer
%%
[~, idx] = sort([transformed.dist_to_gt], 'ascend');
transformed2 = transformed(idx);
%plot([transformed2.dist_to_gt])
%plot(dists(idx))
plot_transforms(transformed2, out_img_cropped, this_img.gt_image);

%% reweighting the final image using the found weightings
temp_dist = [transformed2.dist_to_gt] - min([transformed2.dist_to_gt]);
probs =  exp(-temp_dist/50);
probs = probs / sum(probs);
to_use = probs > 0.01;
mask_stack = cell2mat(reshape({transformed2(to_use).cropped_mask}, 1, 1, []));
[~, weighted_stack] = noisy_or(mask_stack, 3, probs(to_use));
summed_stack = sum(weighted_stack, 3);
imagesc2(summed_stack)

% todo - can speed up by only selecting the shapes which are above a
% minimum probabiltiy




