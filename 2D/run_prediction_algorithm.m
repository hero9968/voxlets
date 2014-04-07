% script to run a prediction algorithm

clear
cd ~/projects/shape_sharing/2D
define_params
set_up_predictors
load(paths.split_path, 'split')
addpath src/predict
addpath src/utils
addpath src/external
addpath src/external/hist2
addpath src/external/libicp/matlab/
addpath src/external/findfirst

%%
close all
%profile on
for ii = 5%:length(predictor)
    for jj = 1:length(split.test_data)

        % loading in the depth for this image
        this_filename = split.test_data{jj};
        this_depth_path = fullfile(paths.raytraced, this_filename);
        load([this_depth_path '.mat'], 'this_raytraced_depth');
         

        % making the prediction
        this_prediction = predictor(ii).handle(this_raytraced_depth, jj);
        
        % saving the prediction to disk
        out_file = [split.test_data{jj}, '.png'];
        out_file = [predictor(ii).outpath, out_file];
        imwrite(this_prediction, out_file, 'BitDepth', 8);
        
        jj
        
    end
end

%profile off viewer