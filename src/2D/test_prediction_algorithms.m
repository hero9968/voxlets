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
addpath ../common/

%% loading test data
load(paths.split_path, 'split')
load(paths.test_data, 'test_data')
load(paths.all_images, 'all_images')

%% load predictors
close all
test.idx = 200; % which test image to use

% loading in the test image data
test.depth = test_data(test.idx).depth;
test.segments = test_data(test.idx).segmented;
test.raw_image = all_images{test_data(test.idx).image_idx};
[~, gt_imgs] = rotate_and_raytrace_mask(test.raw_image, test_data(test.idx).angle, 1);
test.ground_truth = im2double(gt_imgs{1});
clear gt_imgs
%transform_image(raw_image, test_data(test_idx).transform.tdata.T);

test.height = size(test.ground_truth, 1);
%predictions = cell(1, length(predictor));
predictions = cell(1, 5);

%%
load(paths.structured_predict_si_model_path, 'model');
params.weights_threshold = 0.99;

%% loop over each prediction algorithm
%profile on
for ii = 1:3
    
    predictor = get_predictor(ii, true, params, paths, model);
    
    %% making the prediction
    tic
    predictions{ii} = predictor.handle(test.depth, test.height, test.segments, test.ground_truth);
    timings(ii) = toc;
    
    done(ii, length(predictor))

end
%profile off viewer

%% plotting all the precitions
[n, m] = best_subplot_dims(length(predictions) + 1);
subplot(n, m, 1);
imagesc(test.ground_truth);
axis image

title('Ground truth')

for ii = 1:length(predictions)

    % making the prediction
    subplot(n, m, ii+1);
    imagesc(predictions{ii});
    axis image
    %title([predictor(ii).shortname, ' - ' num2str(timings(ii))])
    set(gca, 'clim', [0, 1])   
    done(ii, length(predictor))

end

colormap(flipgray)

%profile off viewer
