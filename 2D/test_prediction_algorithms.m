% script to run a prediction algorithm

clear
cd ~/projects/shape_sharing/2D
define_params
load(paths.split_path, 'split')
load(paths.test_data, 'test_data')
addpath src/predict
addpath src/utils
addpath src/external
addpath src/external/hist2
addpath src/external/libicp/matlab/
addpath src/external/findfirst

%% load predictors
predictor = get_predictor(1:8, 1, params, paths);

%%
close all
test_idx = 2000; % which test image to use
predictions = cell(1, length(predictor));

% loading in the test image data
depth = test_data(test_idx).raytraced;
height = size(test_data(test_idx).image, 1);
segments = test_data(test_idx).segmented;
ground_truth = test_data(test_idx).image;

%% loop over each prediction algorithm
for ii = 1:6%length(predictor)

    % making the prediction
    predictions{ii} = predictor(ii).handle(depth, height, segments, test_idx);
   
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
    title(predictor(ii).shortname)
    set(gca, 'clim', [0, 1])   
    done(ii, length(predictor))

end

colormap(flipgray)

%profile off viewer
