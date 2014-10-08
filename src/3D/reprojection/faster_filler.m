% aim is to speed up the fill_depth_colorisation code...
cd ~/projects/shape_sharing/src/3D/reprojection/
addpath('toolbox_nyu_depth_v2/')

%% loading the data
base = '/Users/Michael/projects/shape_sharing/data/bigbird/3m_high_tack_spray_adhesive/';
obj_name = 'NP3_144';
obj = [base, obj_name];

% loading the depth and rgb
depth = h5read([obj, '.h5'], '/depth');
imgDepthAbs = double(depth') / 10000;
imgRgb = imread([obj, '.jpg']);

% loading the intrinsics
K2 = h5read([base, 'calibration.h5'], '/NP3_rgb_K')';
K1 = h5read([base, 'calibration.h5'], '/NP3_depth_K')';

% loading the extrinsics
H2 = h5read([base, 'calibration.h5'], '/H_NP3_from_NP5');
H1 = h5read([base, 'calibration.h5'], '/H_NP3_ir_from_NP5');

%%
gray_image = uint8(ones([size(imgDepthAbs), 3])*128);

badly_filled_depth = fill_depth_cross_bf(imresize(imgRgb, size(imgDepthAbs)), imgDepthAbs);
reproj_rgb = uint8(reproject_rgb_into_depth(imgRgb, badly_filled_depth, K1, K2, H1', H2'));

better_filled_depth = fill_depth_cross_bf(reproj_rgb, imgDepthAbs);
better_filled_depth2 = fill_depth_colorization2(im2double(rgb2gray(reproj_rgb)), imgDepthAbs);
better_reproj_rgb = uint8(reproject_rgb_into_depth(imgRgb, better_filled_depth, K1, K2, H1', H2'));


%%
subplot(221)
imagesc(uint8(reproj_rgb))
axis image
subplot(222)
imagesc(uint8(better_reproj_rgb))
axis image
subplot(223)
imagesc(uint8(even_better_reproj_rgb))
axis image
%%
subplot(221)
imagesc((imgDepthAbs))
axis image
subplot(222)
imagesc((badly_filled_depth))
axis image
subplot(223)
imagesc((better_filled_depth))
axis image
subplot(224)
imagesc((even_better_filled_depth))
axis image


%%
%dbquit
%mex get_vals_mex.cpp
mex inner_loop.cpp

%% cropping the data
clc
%D = double(imgDepthAbs(50:190, 50:190));
D = double(imgDepthAbs);
RGB = imresize(im2double(imgRgb), size(D));
GRAY = rgb2gray(RGB);
%%m

profile on
tic
D2 = fill_depth_colorization3(RGB, D);
toc
profile off viewer
imagesc([D, D2])

% 31*31*9-(3*4*31)+4
