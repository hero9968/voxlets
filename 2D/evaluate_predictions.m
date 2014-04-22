% script to evaluate a set of predictions.
% Given a folder of ground truth images and a folder of predicted images ?
% This script compares the two and gives some kind of feedback on their
% similarity or differences

clear
cd ~/projects/shape_sharing/2D
define_params
set_up_predictors
load(paths.split_path, 'split')
length_test_data = length(test_data);

%% loading in the ground truth files
all_GT = {test_data.image};

%% saving to disk results for aach fo thw algorithms
% Need to run this for each set of results after they have been generated,
% before we can plot them etc.
%
%profile on
for ii = [1]%1:length(predictor)
    
    predicted_path = predictor(ii).outpath;
    all_pred = cell(length_test_data, 1);
    
    savepath = [predictor(ii).outpath, 'combined.mat'];
    load(savepath, 'all_predictions');

    for jj = 1:length_test_data %length(split.test_data)

        % loading in the preixted image
        this_predicted_image = single(all_predictions{jj});
        this_predicted_image(this_predicted_image<0) = 0;
        this_predicted_image(this_predicted_image>1) = 1;
        
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


