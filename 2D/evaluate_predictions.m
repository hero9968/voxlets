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
end


%%
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
        pred(ii).ssd(jj) = sqrt(sum(diffs.^2)) / length(diffs);
        pred(ii).sd(jj) = sum(abs(diffs)) / length(diffs);
        
        pred(ii).image_auc(jj) = plot_roc(all_GT{jj}, this_predicted_image(:));

        % saving the images    
        all_pred{jj} = this_predicted_image(:);
        
        done(jj, length_test_data, 50);
    end

    % now some kind of ROC curve?
    all_GT2 = single(cell2mat(all_GT));
    all_pred2 = single(cell2mat(all_pred));

    [pred(ii).auc, pred(ii).tpr, pred(ii).fpr, pred(ii).thresh] = plot_roc(all_GT2, all_pred2);
    %hold on
    ii

end

%% plotting ROC curves
cols = {'r-', 'b-', 'g-', 'k-', 'r:'};
for ii = 1:length(predictor)
    plot_roc_curve(pred(ii).tpr, pred(ii).fpr, cols{ii}, pred(ii).thresh); 
    hold on
end
legend({predictor.nicename}, 'Location', 'SouthEast')
hold off
set(findall(gcf,'type','text'),'fontSize',18)

%% finding best and worst matches
plotting = 0;
saving = 1;

this_alg = 1;
[~, idx] = sort(pred(this_alg).ssd, 'ascend');
predicted_path = predictor(this_alg).outpath;

foldername = sprintf('../media/best_worst/%s', predictor(this_alg).name);
if saving && ~exist(foldername, 'dir')
    mkdir(foldername)
end

%for ii = 
num_each_side = 20;
for ii = [1:num_each_side, length(idx):-1:(length(idx)-num_each_side)]
    
    this_idx = idx(ii);
    
    this_predicted_path = fullfile(predicted_path, [split.test_data{this_idx}, '.png']);
    this_predicted_image = single(imread(this_predicted_path));
    this_predicted_image = this_predicted_image / max(this_predicted_image(:));

    this_GT_path = fullfile(ground_truth_path, [split.test_data{this_idx}, '.gif']);
    this_GT_image = imread(this_GT_path);

    % plot the gt image next to the predicted
    %entries = sum(this_GT_image, 2);
    height = size(this_GT_image, 1); %length(entries) - findfirst(flipud(entries), 1, 1, 'first') + 20;
    

    subplot(131)
    imagesc(this_GT_image(1:height, :)); 
    axis image off; 
    colormap(flipgray)
    title('Ground truth')
    
    subplot(132)
    imagesc(fill_grid_from_depth(raytrace_2d(this_GT_image), height, 0.5))
    axis image off
    title('Input')
    subplot(133)
    imagesc(this_predicted_image(1:height, :)); axis image off; 
    
    colormap(flipgray)
    title([num2str(ii) ' of ' num2str(length(idx))]);
    %title(num2str(pred(this_alg).sd(this_idx)));
    
    if plotting
        drawnow
        pause(1.5)
    end
    
    if saving
        filename = sprintf('../media/best_worst/%s/%03d.png', predictor(this_alg).name, ii);
        print(gcf, filename, '-dpng')
    end
    
end

