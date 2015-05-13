% script to run a prediction algorithm

clear
cd ~/projects/shape_sharing/2D
define_params
addpath(genpath('./src'))
num_predictors = 3;

%% getting model etc
load(paths.test_data_subset, 'test_data')
%{
load(paths.test_data, 'test_data')
test_subset = randperm(length(test_data), 25);
test_data = test_data(test_subset);
save(paths.test_data_subset, 'test_data')
%}

load(paths.all_images, 'all_images')
load(paths.structured_predict_si_model_path, 'model');
params.weights_threshold = 0.99;

%%
close all

% loop over each prediction algorithm
for ii = 1%:num_predictors
    
    % loading this predictor, including the handle to run the prediction
    predictor = get_predictor(ii, 1, params, paths, model);
    
    all_predictions = cell(1, length(test_data));
     
    % loop over each test image
    for jj = 1:length(test_data)

        % loading in the depth for this image
        depth = test_data(jj).depth;
        segments = test_data(jj).segmented;

        raw_image = all_images{test_data(jj).image_idx};
        [~, gt_imgs] = rotate_and_raytrace_mask(raw_image, test_data(jj).angle, 1);
        ground_truth = im2double(gt_imgs{1});
        clear gt_imgs;
        
        height = size(ground_truth, 1);
         
        % making the prediction
        this_prediction = predictor.handle(depth, height, segments, ground_truth);
        all_predictions{jj} = this_prediction;
        
        done(jj, length(test_data))
        
    end
    
    savepath = [predictor.outpath, 'combined.mat'];
    save(savepath, 'all_predictions');
end

%profile off viewer
