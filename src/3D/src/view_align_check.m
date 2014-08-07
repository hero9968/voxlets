% a file to load in some of the rendered views and to plot them all
% together to check that they all align
cd ~/projects/shape_sharing/src/3D/src
clear
addpath plotting/
addpath utils/
addpath transformations/
run ../define_params_3d.m

%% setting up the paths
num = 100;
model = '6d9b13361790d04d457ba044c28858b1';%params.model_filelist{num};

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
all_xyz_trans = cell(1, 42);

for ii = 1:42
    
    % extracting the xyz points
    this_depth = depths{ii};
    this_xyz = reproject_depth(this_depth, params.half_intrinsics, max_depth);
    this_xyz(:, 2) = -this_xyz(:, 2);
    this_xyz(:, 3) = -this_xyz(:, 3);
    all_xyz{ii} = this_xyz;
    
    % extracting the rotation matrix
    rot_name = sprintf('/Users/Michael/projects/shape_sharing/data/3D/basis_models/halo/mat_%d.csv', ii);
    T = csvread(rot_name);
    
    % applying the suitable transformation to get back into canonical view
    this_xyz_trans = apply_transformation_3d(this_xyz, (T));
    all_xyz_trans{ii} = this_xyz_trans;
    
    % adding to the plot in a different colour
    plot3d(this_xyz_trans, cols(ii));
    hold on
    
end

hold off

%% now loading in the voxel grid for this modelclf
%voxel_filename = './voxelisation/clement_carving/temp.mat';%sprintf('/Users/Michael/projects/shape_sharing/data/3D/basis_models/voxelised/%s.mat', model);
voxel_filename = ['~/projects/shape_sharing/data/3D/basis_models/voxelised/' model '.mat'];
vox_struct = load(voxel_filename);
V = full_3d(150*[1, 1, 1], vox_struct.sparse_volume);
V = permute(V, [2, 1, 3]);

%%
%
clf
R = [-vox_struct.size, vox_struct.size];
vol3d('CData', double(V), 'XData', R, 'YData', R, 'ZData', R)
axis image

hold on
for ii = 1:5:42
    plot3d(all_xyz_trans{ii}, cols(ii));
end
hold off

view(10, -4)

%% alternate viewing system, showing voxels with with proper transformation
clf
height = -0.2;
thresh = 40.01;
scale = 1/vox_struct.size;
T_vox = [0 scale 0 -0.5-0.0025; 
        scale 0 0 -0.5-0.0025; 
        0 0 scale -0.5; 
        0 0 0 1];
cols = 'rgykcrgykcrgykcrgykcrgykcrgykcrgykcrgykcrgykcrgykcrgykcrgykc';
[inds] = find(V);
[i, j, k] = ind2sub(size(V), inds);
trans_vox = apply_transformation_3d([i,j,k], T_vox);
to_plot = abs(trans_vox(:, 3) - height) < thresh;
plot3(trans_vox(to_plot, 1), trans_vox(to_plot, 2), trans_vox(to_plot, 3), '.', 'markersize', 10)
axis image
%%
hold on
count = 1;
for ii = 1:10:42
    to_plt = abs(all_xyz_trans{ii}(:, 3) - height) < thresh;
    plot3d(all_xyz_trans{ii}(to_plt, :), cols(count));
    count = count + 1
end
hold off

view(0, 90)

%% plotting objects in their natural view with the voxels transformed to them!
count = 1;
for ii = 1:4%42
    
    subplot(2, 2, count);
    count = count + 1;
    plot3d(all_xyz{ii}, 'g')
    view(0, 90)
    
    % transform voxels
    rot_name = sprintf('/Users/Michael/projects/shape_sharing/data/3D/basis_models/halo/mat_%d.csv', ii);
    T = csvread(rot_name);
    temp_vox = apply_transformation_3d(trans_vox, inv(T));
    
    hold on
    plot3d(temp_vox)
    hold off
    
    % adding voxels in
end




