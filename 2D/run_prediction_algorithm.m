% script to run a prediction algorithm

clear
cd ~/projects/shape_sharing/2D
define_params
load(paths.split_path, 'split')
load(paths.test_data, 'test_data')
addpath src/predict
addpath src/utils
addpath src/external
addpath src/transformations/
addpath src/external/hist2
addpath src/external/libicp/matlab/
addpath src/external/findfirst
num_predictors = 8;

%%
close all

% loop over each prediction algorithm
for ii = 1:num_predictors
    
    % loading this predictor, including the handle to run the prediction
    predictor = get_predictor(ii, 1, params, paths);
    
    all_predictions = cell(1, length(test_data));
     
    % loop over each test image
    for jj = 1:length(test_data)

        % loading in the depth for this image
        depth = test_data(jj).raytraced;
        segments = test_data(jj).segmented;
        ground_truth = test_data(jj).image;
        height = size(test_data(jj).image, 1);
         
        % making the prediction
        this_prediction = predictor.handle(depth, height, segments, jj);
        all_predictions{jj} = this_prediction;
        
        done(jj, length(split.test_data))
        
    end
    
    savepath = [predictor.outpath, 'combined.mat'];
    save(savepath, 'all_predictions');
end

%profile off viewer
