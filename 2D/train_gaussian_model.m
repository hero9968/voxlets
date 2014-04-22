% a script to train a model from the training data...

cd ~/projects/shape_sharing/2D
clear
define_params
addpath src/predict
addpath src/external/
addpath src/external/findfirst

%% loading in all depths and shapes from disk...
load(paths.all_images, 'all_images')
load(paths.train_data, 'train_data')
load(paths.test_data, 'test_data')

%% now compute the model
training_images = {train_data.image};
all_depths = {train_data.raytraced};
model = gaussian_model_train(training_images, all_depths, params);

%% plotting the model distributions
subplot(211); plot(model.means)
subplot(212); plot(model.stds)

%% save the model
if ~exist(paths.models_path, 'dir')
    mkdir(paths.models_path)
end
save(paths.gaussian_predict_model_path, 'model');

%% do a preidction
num = 461;
pred = gaussian_model_predict(model, test_data(num).raytraced, size(test_data(num).image, 1));
clf
subplot(121)
imagesc2(test_data(num).image); 
axis image
subplot(122)
imagesc2(pred); 
axis image
colormap(flipgray)

%% plotting some other stuff
all_depths = reshape(cell2mat(model.per_bin_depths(10, :)), 1, []);
clf
hist(all_depths, 100)
xlabel('Depth (as a fraction of image width)')
ylabel('Frequency')





