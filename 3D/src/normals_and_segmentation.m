% a script to load in a depth image, convert to xyz, compute normals and segment
clear
cd ~/projects/shape_sharing/3D/src/
addpath plotting/
addpath features/
addpath ./file_io/matpcl/
addpath ../../common/
addpath transformations/
addpath ../../2D/src/segment/
run ../define_params_3d.m

%% loading in depth
%{
num = 1;
render_idx = 10;
model = params.model_filelist{num};
depth_name = sprintf(paths.basis_models.rendered, model, render_idx);
load(depth_name, 'depth');
depth(abs(depth-3) < 0.001) = nan;
%}
%% Loading in real depth image!
%{
depth_path = '/Users/Michael/data/others_data/rgbd-scenes/desk/desk_1/desk_1_50_depth.png';
rgb_path = '/Users/Michael/data/others_data/rgbd-scenes/desk/desk_1/desk_1_50.png';
depth = double(imread(depth_path))/1000;
depth(depth==0) = nan;
%imagesc(depth)
cloud.rgb = imread(rgb_path);
%}

%% project depth and compute normals
cloud.depth = depth;
cloud.xyz = reproject_depth(cloud.depth, params.rgbddataset_intrinsics);

%% loading in some of the ECCV dataset
clear cloud
filepath = '/Users/Michael/data/others_data/ECCV_dataset/pcd_files/frame_20111220T111153.549117.pcd';
P = loadpcd(filepath);
cloud.xyz = P(:, :, 1:3);
cloud.xyz = reshape(permute(cloud.xyz, [3, 1, 2]), 3, [])';
cloud.rgb = P(:, :, 4:6);
image(cloud.rgb); axis image
cloud.depth = reshape(P(:, :, 3), [480, 640]);
%imagesc(cloud.depth); axis image

%%
[cloud.normals, cloud.curvature] = normals_wrapper(cloud.xyz, 'knn', 50);
%plot_normals(cloud.xyz, cloud.normals, 0.05)

%% now do some kind of segmentation...
opts.min_cluster_size = 500;
opts.max_cluster_size = 1e6;
opts.num_neighbours = 50;
opts.smoothness_threshold = (7.0 / 180.0) * pi;
opts.curvature_threshold = 1.0;
opts.overlap_threshold = 0.9;

%% single segmentation
[idx] = segment_wrapper(cloud, opts);
nansum(idx)
imagesc(reshape(idx, 480, 640))

%% running segment soup algorithm
[idxs, idxs_without_nans] = segment_soup_3d(cloud, opts);

%% plotting
plot_segment_soup_3d(cloud.depth, idxs);

%% attempt to read a pcd file
