% a script to plot nicely a specific prediction result

clear
cd ~/projects/shape_sharing/2D/src
run ../define_params
run ../set_up_predictors
load(paths.split_path, 'split')
load(paths.test_data)
addpath src/utils

%%
%num = 114;
num = num + 1
predictor_idx = 1;
this_img = test_data.images{num};
this_depth = test_data.depths{num};
pred_filename = [split.test_data{num}, '.png'];
pred_filename = [predictor(predictor_idx).outpath, pred_filename];
this_predicted = single(imread(pred_filename)) /256;

subplot(131)
imagesc(this_img)
axis image

subplot(132)
imagesc(fill_grid_from_depth(this_depth, size(this_img, 1), 0.4))
axis image

subplot(133)
imagesc(this_predicted)
set(gca, 'clim', [0, 1])
axis image

colormap(flipgray)







