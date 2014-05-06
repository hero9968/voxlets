% a script to load in a depth image, convert to xyz, compute normals and
% segment
clear
cd ~/projects/shape_sharing/3D/features/
addpath ../plotting/
addpath ../transformations/
run ../define_params_3d.m
num = 100;
model = params.model_filelist{num};

%% loading in depth
ii = 10;
depth_name = sprintf(paths.basis_models.rendered, model, ii);
load(depth_name, 'depth');
depth(abs(depth-3) < 0.001) = nan;

%% converting to xyz
cloud.xyz = reproject_depth(depth, params.half_intrinsics);

%% compute normals
[cloud.normals, T] = normals_wrapper(cloud.xyz, 'knn', 150);

%% plot normals
plot_normals(cloud.xyz, cloud.normals, 0.05)

%% now do some kind of segmentation...
idx = segment_wrapper(cloud.xyz, cloud.normals);
nansum(idx)

imagesc(reshape(idx, 240, 320))