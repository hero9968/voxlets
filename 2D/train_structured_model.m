% a script to train a model from the training data...

cd ~/projects/shape_sharing/2D
clear
define_params
addpath src/predict
addpath src/utils
addpath src/external/
addpath src/external/hist2
addpath src/external/findfirst
addpath src/external/libicp/matlab

%% loading in all depths and shapes from disk...
load(paths.train_data, 'train_data')
load(paths.test_data, 'test_data')

%% now compute the model
params.scale_invariant = true;
params.sd_angles = 1;
model = train_fitting_model({train_data.image}, {train_data.raytraced}, params);
model.images = cellfun(@(x)(uint8(x)), {train_data.image}, 'UniformOutput', 0);
model.depths = {train_data.raytraced};
all_dists = cell2mat(model.shape_dists);
imagesc(all_dists)

%% save the model
save(paths.structured_predict_si_model_path, 'model');

%%
clf
num = 3940;
params.aggregating = 1;
params.num_proposals= 30;
params.plotting.plot_transforms = 1;
params.plotting.plot_matches = 0;
%num = num+500;

for ii = 1:3
    subplot(1, 4,ii); 
    combine_mask_and_depth(test_data(num).image, test_data(num).raytraced)
    width = length(test_data(num).raytraced);
    set(gca, 'xlim', round([-width/4, 1.25*width]));
    set(gca, 'ylim',round([-width/4, 1.25*width]));
end

height = size(test_data(num).image, 1);
stacked_image = test_fitting_model(model, test_data(num).raytraced, height, params);
subplot(1, 4, 4);
imagesc(stacked_image); axis image


%% showing the closest matching features to the input image
clf
load(paths.structured_predict_si_model_path, 'model');

params.aggregating = 1;
params.plotting.plot_matches = 1;
params.plotting.num_matches = 25;
params.plotting.plot_transforms = 0;

% finding the nearest neighbours
height = size(test_data(num).image, 1);
S = test_fitting_model(model, test_data(num).raytraced, height, params);

% showing image of the input data and ground truth
subplot(3, 4, 3*4);
combine_mask_and_depth(test_data(num).image, test_data(num).raytraced)
title('Ground truth')


