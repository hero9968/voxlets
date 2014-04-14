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
params.scale_invariant = false;
params.sd_angles = 0;
model = train_fitting_model(train_data.images, train_data.depths, params);
model.images = train_data.images;
model.depths = train_data.depths;
all_dists = cell2mat(model.shape_dists);
imagesc(all_dists)

%% save the model
save(paths.structured_predict_model_path, 'model');

%% now compute the model
params.scale_invariant = true;
params.sd_angles = 1;
model = train_fitting_model(train_data.images, train_data.depths, params);
model.images = train_data.images;
model.depths = train_data.depths;
all_dists = cell2mat(model.shape_dists);
imagesc(all_dists)

%% save the model
save(paths.structured_predict_si_model_path, 'model');

%%

%num = 140;
params.aggregating = 1;
params.num_proposals= 10;
num = 415
%num = num+1;
for ii = 1:3
    subplot(1, 4,ii); 
    combine_mask_and_depth(test_data.images{num}, test_data.depths{num})
    width = length(test_data.depths{num})
    set(gca, 'xlim', round([-width/4, 1.25*width]));
    %set(gca, 'ylim',round([-width/4, 1.25*width]));
end

stacked_image = test_fitting_model(model, test_data.depths{num}, test_data.heights(num), params);
subplot(1, 4, 4);
imagesc(stacked_image); axis image

%%
clf
stacked_image = test_fitting_model(model, test_data.depths{num}, test_data.heights(num), params);

%% showing the closest matching features to the input image
clf
%num = num-1;
params.aggregating = 1;
subplot(3, 4, 1);
combine_mask_and_depth(test_data.images{num}, test_data.depths{num})

load(paths.structured_predict_si_model_path, 'model');
S = test_fitting_model(model, test_data.depths{num}, test_data.heights(num), params);
subplot(3, 4, 2);
imagesc(S)
axis image
%load(paths.structured_predict_model_path, 'model');
%test_fitting_model(model, test_data.depths{num}, test_data.heights(num), params);


%%

for ii = 1:25
    name = sprintf('../data/2D_shapes/rotated/01_%02d_mask.gif', ii);
    T = imread(name);
    subaxis(5, 5, ii)
    imagesc(T)
    axis image off
end





