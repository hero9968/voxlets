% a file to load in some of the rendered views and to plot them all
% together to check that they all align
cd ~/projects/shape_sharing/3D/model_render/
clear
addpath ../plotting/
addpath src
addpath ../transformations/
run ../define_params_3d.m

%% setting up the paths
num = 10;
model = params.model_filelist{num};

%% loading in the views and plotting
num_views = 42;
max_depth = 3;
depths = cell(1, num_views);

for ii = 1:num_views
    % loading the depth
    depth_name = sprintf(paths.basis_models.rendered, model, ii);
    load(depth_name, 'depth');
    
    % plotting depth image
    subplot(6, 7, ii);
    render_depth(depth, max_depth)
    
    depths{ii} = depth;
end
colormap(hot)

%% projecting each depth image into 3d and 3d plotting...
%intrinsics = 
clf
cols = 'rgbkcymrgbkcymrgbkcymrgbkcymrgbkcymrgbkcymrgbkcymrgbkcymrgbkcymrgbkcymrgbkcymrgbkcymrgbkcymrgbkcymrgbkcym';

for ii = 1:3:42
    
    % extracting the xyz points
    this_depth = depths{ii};
    this_xyz = reproject_depth(this_depth, params.half_intrinsics, max_depth);
    this_xyz(:, 2) = -this_xyz(:, 2);
    this_xyz(:, 3) = -this_xyz(:, 3);
    
    
    % extracting the rotation matrix
    rot_name = sprintf('/Users/Michael/projects/shape_sharing/data/3D/basis_models/halo/mat_%d.csv', ii);
    T = csvread(rot_name);
    
    % applying the suitable transformation to get back into canonical view
    this_xyz_trans = apply_transformation_3d(this_xyz, (T));
    
    % adding to the plot in a different colour
    plot3d(this_xyz_trans, cols(ii));
    hold on
    
end

hold off




