% script to evaluate a set of predictions.
% Given a folder of ground truth images and a folder of predicted images ?
% This script compares the two and gives some kind of feedback on their
% similarity or differences

clear
cd ~/projects/shape_sharing/2D/testing
run ../define_params.m
load(paths.split_path, 'split')
length_test_data = length(split.test_data);
cd ~/projects/shape_sharing/2D/testing

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
end


%%
for ii = 1:3
    
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
        pred(ii).ssd(jj) = sqrt(sum(diffs.^2)) / length(diffs);
        pred(ii).sd(jj) = sum(abs(diffs)) / length(diffs);

        % saving the images    
        all_pred{jj} = this_predicted_image(:);
        jj
    end

    % now some kind of ROC curve?
    all_GT2 = single(cell2mat(all_GT));
    all_pred2 = single(cell2mat(all_pred));

    [pred(ii).auc, pred(ii).tpr, pred(ii).fpr] = plot_roc(all_GT2, all_pred2);
    %hold on
    ii

end

%% plotting ROC curves
cols = {'r-', 'b:', 'g--'};
for ii = 1:3
    plot(pred(ii).tpr, pred(ii).fpr, cols{ii}); 
    hold on
end
hold off
axis image
xlabel('FPR'); ylabel('TPR')
legend({predictor.nicename}, 'Location', 'SouthEast')


%% finding best and worst matches

this_alg = 3;
[~, idx] = sort(pred(this_alg).ssd, 'ascend');

for ii = 15%10:20
    this_idx = idx(ii);
    
    this_predicted_path = fullfile(predicted_path, [split.test_data{this_idx}, '.png']);
    this_predicted_image = single(imread(this_predicted_path));
    this_predicted_image = this_predicted_image / max(this_predicted_image(:));

    this_GT_path = fullfile(ground_truth_path, [split.test_data{this_idx}, '.gif']);
    this_GT_image = imread(this_GT_path);

    % plot the gt image next to the predicted
    subplot(131)
    imagesc(this_GT_image(1:100, :)); axis image off; colormap(gray)
    subplot(132)
    imagesc(fill_grid_from_depth(raytrace_2d(this_GT_image), 100, 0.5))
    axis image off
    subplot(133)
    imagesc(this_predicted_image(1:100, :)); axis image off; colormap(gray)
    title(num2str(pred(this_alg).ssd(this_idx)));
    pause(1.5)
    
    
end

