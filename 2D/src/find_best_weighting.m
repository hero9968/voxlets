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
num = 122;
depth = test_data.depths{num};
transforms = propose_transforms(model, depth, params);
[out_img, out_img_cropped, transformed] = aggregate_masks(transforms, params.im_height, depth);

subplot(4, 5, 1); 
imagesc(test_data.images{num})
axis image

for ii = 1:18; 
    subplot(4, 5, ii+1); 
    imagesc(transformed(ii).masks); 
    axis image
end

subplot(4, 5, 20); 
%clf
imagesc(out_img)
axis image
%T = sum(cell2mat(reshape(transformed.imgs, 1, 1, [])), 3);
%clf
%%
clf
[X, Y] = find(edge(test_data.images{num}));
x_loc = transformed(1).padding; %find(transformed.x_range==1);
y_loc = transformed(1).padding; %find(transformed.y_range==1);
xy = apply_transformation_2d([X'; Y'], [1, 0, y_loc; 0, 1, x_loc; 0 0 1]);

imagesc(out_img);
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
