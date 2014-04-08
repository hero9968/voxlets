% a script to train a model from the training data...

cd ~/projects/shape_sharing/2D
clear
define_params
addpath src/predict
addpath src/external/
addpath src/external/findfirst

%% loading in all depths and shapes from disk...
load(paths.train_data, 'train_data')
load(paths.test_data, 'test_data')

%% now compute the model
model = gaussian_model_train(train_data.images, train_data.depths, params);
subplot(211); plot(model.means)
subplot(212); plot(model.stds)

%% save the model
if ~exist(paths.models_path, 'dir')
    mkdir(paths.models_path)
end
save(paths.gaussian_predict_model_path, 'model');

%% do a preidction
num = 361;
pred = gaussian_model_predict(model, test_data.depths{num}, test_data.heights(num));
clf
subplot(121)
imagesc2(test_data.images{num}); 
axis image
subplot(122)
imagesc2(pred); 
axis image
colormap(flipgray)





