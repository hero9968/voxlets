% script to run a prediction algorithm

clear
cd ~/projects/shape_sharing/2D
define_params
set_up_predictors
load(paths.split_path, 'split')
load(paths.test_data)
addpath src/predict
addpath src/utils
addpath src/external
addpath src/external/hist2
addpath src/external/libicp/matlab/
addpath src/external/findfirst

%%
close all

% loop over each prediction algorithm
for ii = 6 %:length(predictor)
    
    all_predictions = cell(1, length(split.test_data));
    
    % loop over each test image
    for jj = 1:length(split.test_data)

        % loading in the depth for this image
        this_filename = split.test_data{jj};
        this_depth_path = fullfile(paths.raytraced, this_filename);
        height = test_data.heights(jj);
        load([this_depth_path '.mat'], 'this_raytraced_depth');
         
        % making the prediction
        this_prediction = predictor(ii).handle(this_raytraced_depth, height, jj);
        
        % saving the prediction to disk
        out_file = [split.test_data{jj}, '.png'];
        out_file = [predictor(ii).outpath, out_file];
        imwrite(this_prediction, out_file, 'BitDepth', 8);
        
        
        all_predictions{jj} = this_prediction;
        done(jj, length(split.test_data))
        
    end
    
    savepath = [predictor(ii).outpath, 'combined.mat'];
    save(savepath, 'all_predictions');
end

%profile off viewer


%% temp script to read in images and save them to mat files
close all

% loop over each prediction algorithm
for ii = 6 %:length(predictor)
    
    all_predictions = cell(1, length(split.test_data));
    
    % loop over each test image
    for jj = 1:length(split.test_data)
        
        % saving the prediction to cell array
        out_file = [split.test_data{jj}, '.png'];
        out_file = [predictor(ii).outpath, out_file];
        this_prediction = imread(out_file);
                
        all_predictions{jj} = this_prediction;
        done(jj, length(split.test_data))
        
    end
    
    savepath = [predictor(ii).outpath, 'combined.mat'];
    save(savepath, 'all_predictions');
end

%profile off viewer