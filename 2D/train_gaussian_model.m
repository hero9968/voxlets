% a script to train a model from the training data...

cd ~/projects/shape_sharing/2D
clear
define_params
addpath src/predict
addpath src/external/
addpath src/external/findfirst
addpath src/transformations/

%% loading in all depths and shapes from disk...
load(paths.all_images, 'all_images')
load(paths.train_data, 'train_data')
load(paths.test_data, 'test_data')

%% now compute the model
train_data_subset = train_data(randperm(length(train_data), 2000));
model = gaussian_model_train(all_images, train_data_subset, params);

%% plotting the model distributions
subplot(211); plot(model.means)
subplot(212); plot(model.stds)

%% save the model
if ~exist(paths.models_path, 'dir')
    mkdir(paths.models_path)
end
save(paths.gaussian_predict_model_path, 'model');

%% do a preidction
num = 2461;
test_depth = test_data(num).depth;
pred = gaussian_model_predict(model, test_depth, length(test_depth));
clf
subplot(121)
test_image_idx = test_data(num).image_idx;
test_transform = test_data(num).transform;
depth_width = length(test_depth);
gt_image = imtransform(all_images{test_image_idx}, test_transform, ...
    'XYScale',1,'xdata', [1, depth_width], 'ydata', [1, depth_width]);

% plotting gt
imagesc2(gt_image); 
hold on
plot(1:length(test_depth), test_depth);
hold off
axis image

% plotting prediction
subplot(122)
imagesc2(pred); 
axis image
colormap(flipgray)

%% plotting distribution along one strip, to see if roughly gaussian
strip_number = 10;
all_strips = model.per_bin_depths(strip_number, :)';
to_remove = cellfun(@isempty, all_strips);
all_depths = reshape(cell2mat(all_strips(~to_remove)), 1, []);
clf
hist(all_depths, 100)
xlabel('Depth (as a fraction of image width)')
ylabel('Frequency')





