% script to make images for a paper

% set paths
addpath(genpath('../'))
load('../edges_and_compass/data/modelNyud.mat')
addpath('/Users/Michael/builds/research_code/structured_edge_detection/release')
addpath(genpath('/Users/Michael/projects/shape_sharing/src/matlab/toolbox'))

%% Load in a single kinect image
image_path = '/Volumes/HDD/data/others_data/RGBD_datasets/rgbd_scenes2//rgbd-scenes-v2/imgs/scene_02/';
image_name = '00401';
rgb_path = [image_path, image_name, '-color.png'];
depth_path = [image_path, image_name, '-depth.png'];

im.rgb = imread(rgb_path);
im.gray = im2double(rgb2gray(im.rgb));
im.depth = double(imread(depth_path))/10000;
im.smoothed_depth = fill_depth_colorization_ft_mex(im.gray, im.depth);

%% doing the edges
im.struct_edges = edgesDetect(im.smoothed_depth, model);
im.struct_edges_canny = depth_canny(im.struct_edges, im.smoothed_depth);
se = strel('disk',1);
im.struct_edges_canny = imdilate(im.struct_edges_canny, se)

%% preparing for plotting...
im.dmin = nanmin(im.depth(im.depth(:)~=0))
im.dmax = nanmax(im.depth(:))

%%
temp = convert_to_jet(im.depth);
imshow(temp)

%% Plot all

subplot(231)
imshow(im.rgb)
imwrite(im.rgb, 'preprocess_a.png')

subplot(232)
h = imagesc(im.depth);
set(h, 'AlphaData', im.depth~=0)
set(gca, 'clim', [im.dmin, im.dmax])
axis image off
temp_depth = im.depth;
temp_depth(temp_depth == 0) = nan;
imwrite(convert_to_jet(temp_depth), 'preprocess_b.png')

subplot(233)
imagesc(im.smoothed_depth)
set(gca, 'clim', [im.dmin, im.dmax])
axis image off
imwrite(convert_to_jet(im.smoothed_depth), 'preprocess_c.png')

subplot(234)
imshow(im.struct_edges)
imwrite(im.struct_edges, 'preprocess_d.png')

subplot(235)
imshow(im.struct_edges_canny)
imwrite(im.struct_edges_canny, 'preprocess_e.png')

colormap(jet)
axis image