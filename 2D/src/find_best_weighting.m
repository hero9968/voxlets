% the aim of this script is to find a good weighting for the basis
% functions, given that we know the ground truth mask.

cd ~/projects/shape_sharing/2D/src
clear
run ../define_params
addpath predict
addpath utils
addpath external/
addpath external/hist2
addpath external/findfirst
addpath external/libicp/matlab

%% loading in model and test data
load(paths.test_data, 'test_data')
load(paths.structured_predict_si_model_path, 'model');

%% 
clf
num = 332;
depth = test_data.depths{num};
transforms = propose_transforms(model, depth, params);
[out_img, out_img_cropped, transformed] = aggregate_masks(transforms, params.im_height, depth);
plot_transforms(transformed, out_img_cropped, test_data.images{num});

%% Here will try to optimise for the weights
% want to find the weights that minimise the sum of squared errors over the
% hidden part of the image
gt_img = single(test_data.images{num});
mask_stack = single(cell2mat(reshape({transformed.cropped_mask}, 1, 1, [])));

[weights, other] = find_optimal_weights(depth, mask_stack, gt_img);
%
clf; 
subplot(121)
imagesc(gt_img(1:other.height, :))
axis image

subplot(122)
imagesc(other.final_image)
axis image
set(gca, 'clim', [0, 1])
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
