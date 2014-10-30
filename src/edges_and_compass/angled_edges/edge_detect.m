
%% set up
cd ~/projects/shape_sharing/src/edges_and_compass/angled_edges/
addpath('/Users/Michael/builds/research_code/structured_edge_detection/release')
addpath(genpath('/Users/Michael/projects/shape_sharing/src/matlab/toolbox'))
load('../data/modelNyud.mat')

%% load the data to test on
if true
    % loading rgbd scenes
    image_path = '/Volumes/HDD/data/others_data/RGBD_datasets/rgbd_scenes2//rgbd-scenes-v2/imgs/scene_02/';
    image_name = '00401';
    rgb_path = [image_path, image_name, '-color.png'];
    depth_path = [image_path, image_name, '-depth.png'];

    C.rgb = imread(rgb_path);
    C.gray = im2double(rgb2gray(C.rgb));
    C.depth = double(imread(depth_path))/10000;
    C.smoothed_depth = fill_depth_colorization_ft_mex(C.gray, C.depth);
else
    % loading bigbird
    modelname = 'nutrigrain_apple_cinnamon';
    view = 'NP2_24';
    T = load(['~/projects/shape_sharing/data/bigbird_cropped/', modelname, '/' view '.mat'])
    
end
    
%% doing structured edge detection
structured_edge = edgesDetect(C.smoothed_depth, model);

%% canny edge filtering
canny_edge = depth_canny(structured_edge, C.smoothed_depth);

%% plotting the result
H1 = subplot(222); imagesc(C.smoothed_depth);  axis image
subplot(221); imagesc(C.depth); axis image; set(gca, 'clim', get(H1, 'clim'))
subplot(223); imagesc(structured_edge);  axis image
subplot(224); imshow(repmat(uint8(canny_edge*255), [1, 1, 3])); axis image
colormap(jet)

%% now applying peter's edge angle detection code
[angles, conv, kern] = depth_angles(C.smoothed_depth, canny_edge, 5);
%%
clf
angles(canny_edge==0) = nan;
plot_angles(angles)
%% 

%subplot(131); imagesc(real(conv)); axis image
%subplot(132); imagesc(imag(conv)); axis image
%temp = angle(conv);
%temp(canny_edge==0) = nan;
%clf; h=imagesc(temp); axis image
%set(h, 'alphadata', ~isnan(angles))
%colorbar
