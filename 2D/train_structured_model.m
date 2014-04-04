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
profile on
%num = 140;
params.aggregating = 1;
num = 144
%num = num+1;
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
profile off viewer

%% showing the closest matching features to the input image
clf
num = num-1;
params.aggregating = 0;
subplot(3, 4, 1);
combine_mask_and_depth(test_data.images{num}(1:100, :), test_data.depths{num})

load(paths.structured_predict_si_model_path, 'model');
test_fitting_model(model, test_data.depths{num}, params);
load(paths.structured_predict_model_path, 'model');
test_fitting_model(model, test_data.depths{num}, params);


%% fixing the rotating bug - DONE
clf
num = num+1;
depth = test_data.depths{num};
Y = (double(depth));
X = 1:length(Y);
[~, ~, this_transform_to_origin] = transformation_to_origin_2d(X, Y);
coordinate_frame = [0, 0, 10; 10, 0, 0];
coordinate_frame_trans = apply_transformation_2d(coordinate_frame, this_transform_to_origin);
plot(X, Y, 'o');
hold on
plot(coordinate_frame(1, :), coordinate_frame(2, :), 'r')
plot(coordinate_frame_trans(1, :), coordinate_frame_trans(2, :), 'r')
hold off
axis image





