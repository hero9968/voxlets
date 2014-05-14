% script to test the prediction algorithms, doing one prediction from each

clear
cd ~/projects/shape_sharing/2D
define_params
addpath src/predict
addpath src/utils
addpath src/transformations/
addpath src/external
addpath src/external/hist2
addpath src/external/libicp/matlab/
addpath src/external/findfirst

%% loading test data
load(paths.split_path, 'split')
load(paths.test_data, 'test_data')
load(paths.all_images, 'all_images')

%% load predictors

%%
close all
test_idx = 3000; % which test image to use


% loading in the test image data
depth = test_data(test_idx).depth;
segments = test_data(test_idx).segmented;
raw_image = all_images{test_data(test_idx).image_idx};
[~, gt_imgs] = rotate_and_raytrace_mask(raw_image, test_data(test_idx).angle, 1);
ground_truth = gt_imgs{1};
%transform_image(raw_image, test_data(test_idx).transform.tdata.T);

height = size(ground_truth, 1);
%predictions = cell(1, length(predictor));
predictions = cell(1, 8);

%% loop over each prediction algorithm
for ii = 5%1:length(predictor)

    predictor = get_predictor(ii, 1, params, paths);
    
    %% making the prediction
    tic
  
    
    predictions{ii} = predictor.handle(depth, height, segments, ground_truth);
    timings(ii) = toc;
    
    done(ii, length(predictor))

end


%% plotting all the precitions
[n, m] = best_subplot_dims(length(predictor) + 1);
subplot(n, m, 1);
imagesc(ground_truth);
axis image
title('Ground truth')

for ii = 1:length(predictor)

    % making the prediction
    subplot(n, m, ii+1);
    imagesc(predictions{ii});
    axis image
    title([predictor(ii).shortname, ' - ' num2str(timings(ii))])
    set(gca, 'clim', [0, 1])   
    done(ii, length(predictor))

end

colormap(flipgray)

%profile off viewer
