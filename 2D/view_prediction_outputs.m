% a script to load and view the saved predictions

clear
cd ~/projects/shape_sharing/2D
define_params
set_up_predictors
load(paths.split_path, 'split')
%length_test_data = length(split.test_data);

%% loading in all the results from disk
clear pred

for ii = 1:length(predictor)
    
    predpath = [predictor(ii).outpath, 'evaluation_results.mat'];
    load(predpath, 'this_pred');
    pred(ii) = this_pred;
    
end

%% plotting ROC curves
cols = {'r-', 'b-', 'g-', 'k-', 'c-', 'r:'};
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