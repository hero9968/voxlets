% script to evaluate a set of predictions.
% Given a folder of ground truth images and a folder of predicted images ?
% This script compares the two and gives some kind of feedback on their
% similarity or differences

clear
cd ~/projects/shape_sharing/2D
define_params
set_up_predictors
load(paths.split_path, 'split')
length_test_data = length(split.test_data);

%% loading in the ground truth files
ground_truth_path = paths.rotated;

% get list of GT files
ground_truth_files = dir(ground_truth_path);
to_remove = ismember({ground_truth_files.name}, {'.', '..', '.DS_Store'});
ground_truth_files(to_remove) = [];

% loading in all the GT data
all_GT = cell(length_test_data, 1);
for ii = 1:length_test_data
    
    this_GT_path = fullfile(ground_truth_path, [split.test_data{ii}, '.gif']);
    this_GT_image = imread(this_GT_path);
    all_GT{ii} = this_GT_image(:);
    
    done(ii, length_test_data, 50);
end


%% generating results for the indivudal predictions and saving to disk
% Need to run this for each set of results after they have been generated,
% before we can plot them etc.
%
%profile on
for ii = 1:length(predictor)
    
    predicted_path = predictor(ii).outpath;
    all_pred = cell(length_test_data, 1);

    for jj = 1:length_test_data%length(split.test_data)

        % loading in the preixted image
        this_predicted_path = fullfile(predicted_path, [split.test_data{jj}, '.png']);
        this_predicted_image = single(imread(this_predicted_path));
        this_predicted_image = this_predicted_image / max(this_predicted_image(:));

        % now do the comparison - perhaps try to ignore the pixels that we
        % definately know about?
        %to_use_mask = all_GT{jj}
        diffs = single(all_GT{jj}) - single(this_predicted_image(:));
        this_pred.ssd(jj) = sqrt(sum(diffs.^2)) / length(diffs);
        this_pred.sd(jj) = sum(abs(diffs)) / length(diffs);
        
        this_pred.image_auc(jj) = plot_roc(all_GT{jj}, this_predicted_image(:));

        % saving the images    
        all_pred{jj} = this_predicted_image(:);
        
        done(jj, length_test_data, 50);
    end

    % now some kind of ROC curve?
    all_GT2 = single(cell2mat(all_GT));
    all_pred2 = single(cell2mat(all_pred));

    [this_pred.auc, this_pred.tpr, this_pred.fpr, this_pred.thresh] = plot_roc(all_GT2, all_pred2);
    %hold on
    ii
    
    
    savepath = [predictor(ii).outpath, 'evaluation_results.mat'];
    save(savepath, 'this_pred');
    
end

%profile off viewer


