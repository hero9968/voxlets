% script to run a prediction algorithm
clear
run ../define_params
load(paths.split_path, 'split')
addpath predict


%%
close all
for ii = 2%:length(predictor)
    for jj = 1:length(split.test_data)

        %% loading in the depth for this image
        this_filename = '27_03_mask';%split.test_data{jj};
        this_depth_path = fullfile(paths.raytraced, this_filename);
        this_depth = imread([this_depth_path '.png']);

        % making the prediction
        this_prediction = predictor(ii).handle(this_depth);

        %% saving the prediction to disk
        out_file = [split.test_data{jj}, '.png'];
        out_file = [predictor(ii).outpath, out_file];
        imwrite(this_prediction, out_file);
        
        jj
        
    end
end