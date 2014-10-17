addpath('/Users/Michael/builds/research_code/structured_edge_detection/release')
addpath(genpath('/Users/Michael/projects/shape_sharing/src/matlab/toolbox'))
load('./data/modelNyuRgbd.mat')

%%
stack = cat(3, 0.01*im2double(bb_cropped.rgb), bb_cropped.depth);
structured_edge = edgesDetect(stack, model);
imagesc(structured_edge)
set(gca, 'clim', [0, 1])
axis image

%%
canny_edge = depth_canny(structured_edge, bb_cropped.depth);

%%
%close all
subplot(131); imagesc(bb_cropped.depth); axis image
subplot(132); imagesc(structured_edge); colorbar; axis image
subplot(133); imshow(repmat(uint8(canny_edge*255), [1, 1, 3])); axis image
colormap(jet)

%%
image_path = '/Volumes/HDD/data/others_data/RGBD_datasets/rgbd_scenes2//rgbd-scenes-v2/imgs/scene_02/';
image_name = '00401';
rgb_path = [image_path, image_name, '-color.png'];
depth_path = [image_path, image_name, '-depth.png'];

addpath('~/projects/shape_sharing/src/preprocess/bigbird/crop_smooth/')

C.rgb = imread(rgb_path);
C.gray = im2double(rgb2gray(C.rgb));
C.depth = double(imread(depth_path))/10000;
C.all = cat(3, 0.5*C.rgb, C.depth);
C.edges = edgesDetect(C.all, model);
C.smoothed_depth = fill_depth_colorization_ft_mex(C.gray, C.depth);

%%
stack = cat(3, 0.1*im2double(C.rgb), C.smoothed_depth);
structured_edge = edgesDetect(stack, model);
%structured_edge = structured_edge / max(structured_edge(:));
canny_edge = depth_canny(structured_edge, C.smoothed_depth);

TT = load('./data/modelNyuD.mat');
structured_edge2 = edgesDetect(C.smoothed_depth, TT.model);
canny_edge2 = depth_canny(structured_edge2, C.smoothed_depth);

H1 = subplot(222); imagesc(C.smoothed_depth);  axis image
subplot(221); imagesc(C.depth); axis image; set(gca, 'clim', get(H1, 'clim'))
subplot(223); imagesc(structured_edge);  axis image
subplot(224); imshow(repmat(uint8(canny_edge*255), [1, 1, 3])); axis image
colormap(jet)
%%
close all
imshow(repmat(uint8(canny_edge*255), [1, 1, 3])); axis image

%%
subplot(221)
imagesc(structured_edge); axis image
subplot(222)
imagesc(structured_edge2); axis image
subplot(223)
imagesc(canny_edge); axis image
subplot(224)
imagesc(canny_edge2); axis image
%%
get(H1)

%% want to get the angles of the edges...
[dx, dy] = smoothGradient(C.smoothed_depth, sqrt(2));
angle = mod(atan2(dy, dx), pi);
subplot(121)
imagesc(angle)
axis image
colorbar
%%
temp= double(canny_edge2);
[dx, dy] = smoothGradient(temp, sqrt(2));
[gmag, gdir] = imgradient(dx, dy);%temp, 'roberts');
gdir(~canny_edge2) = nan;
gdir = mod(gdir, 180);
imagesc(gdir)
axis image
% turntable objects to start with

%%

% smooth output

% kinectfusion on sum3d? 

% smoothing on depth image?? Bilateral?


% Fast MRF Optimization with Application to Depth Reconstruction

% canny edge detetor - non-max suppression
% find local peak in 

% graph cuts - remove spurious edges
% slightly fat depth edge, should reinforce where teh gradients are in the
% image. Use internsity image as well 

% spider features for different thresholds

% do on isolated objects to start with 