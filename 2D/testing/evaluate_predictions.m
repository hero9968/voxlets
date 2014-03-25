% script to evaluate a set of predictions.
% Given a folder of ground truth images and a folder of predicted images ?
% This script compares the two and gives some kind of feedback on their
% similarity or differences

clear
run ../define_params.m

% paths input
ground_truth_path = paths.rotated;
predicted_path = predictor(2).outpath;

% get list of GT files
ground_truth_files = dir(ground_truth_path);
to_remove = ismember({ground_truth_files.name}, {'.', '..', '.DS_Store'});
ground_truth_files(to_remove) = [];

load(paths.split_path, 'split')

%%
all_GT = cell(length(split.test_data), 1);
all_pred = cell(length(split.test_data), 1);

for ii = 1:length(split.test_data)
    
    this_GT_path = fullfile(ground_truth_path, [split.test_data{ii}, '.gif']);
    this_predicted_path = fullfile(predicted_path, [split.test_data{ii}, '.png']);
    
    % loading in the two images
    this_GT_image = imread(this_GT_path);
    this_predicted_image = imread(this_predicted_path);
    
    % now do the comparison
    diffs = single(this_GT_image(:)) - single(this_predicted_image(:));
    ssd(ii) = sqrt(sum(diffs.^2));
    sd(ii) = sum(abs(diffs));
    
    % saving the images
    all_GT{ii} = this_GT_image(:);
    all_pred{ii} = this_predicted_image(:);
       
end

%% now some kind of ROC curve?
all_GT2 = single(cell2mat(all_GT));
all_pred2 = single(cell2mat(all_pred));

plot_roc(all_GT2, all_pred2)


