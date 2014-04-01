% compute normals for set of 2D points and plotting them

clear
addpath utils

radius = 4;
hww = 3;

load('~/projects/shape_sharing/data/2D_shapes/raytraced/10_01_mask.mat', 'this_raytraced_depth')
XY = [1:length(this_raytraced_depth); this_raytraced_depth];

norms_r = normals_radius_2d(XY, radius);
norms_n = normals_neighbour_2d(XY, hww);

clf
subplot(211)
plot_normals_2d(XY, norms_r);
subplot(212)
plot_normals_2d(XY, norms_n)