% script to run a prediction algorithm
clear
cd ~/projects/shape_sharing/2D/src
run ../define_params
load(paths.split_path, 'split')
addpath(genpath('src'))


%%
close all
%profile on
for ii = 3%:length(predictor)
    for jj = 18%550:length(split.test_data)

        % loading in the depth for this image
        this_filename = split.test_data{jj};
        this_depth_path = fullfile(paths.raytraced, this_filename);
        load([this_depth_path '.mat'], 'this_raytraced_depth');
         

        % making the prediction
        this_prediction = predictor(ii).handle(this_raytraced_depth);
        
        % saving the prediction to disk
        out_file = [split.test_data{jj}, '.png'];
        out_file = [predictor(ii).outpath, out_file];
        imwrite(this_prediction, out_file, 'BitDepth', 8);
        
        jj
        
    end
end
%profile off viewer