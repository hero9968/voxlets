% script to propose 3D basis shapes for a specific region, and to somehow
% visualise. No transformations or anything clever like that yet...

% a script to load in a depth image, convert to xyz, compute normals and segment
clear
cd ~/projects/shape_sharing/src/3D/src/
addpath(genpath('.'))
addpath(genpath('../../2D/src'))
addpath(genpath('../../common/'))
run ../define_params_3d
load(paths.structured_model_file, 'model')

%% loading in some of the ECCV dataset
cloud = loadpgm_as_cloud('~/projects/shape_sharing/data/3D/scenes/first_few_render_noisy00000.pgm', params.full_intrinsics);
[cloud.normals, cloud.curvature] = normals_wrapper(cloud.xyz, 'knn', 50);

%% running segment soup algorithm
[idxs, idxs_without_nans, probabilities, all_idx] = segment_soup_3d(cloud, params.segment_soup);

%% plotting segments
plot_segment_soup_3d(cloud.rgb.^0.2, idxs, probabilities);

%% Choosing a segment and computing the feature vector
seg_index = 2;
segment = extract_segment(cloud, idxs(:, seg_index), params);

% Find closest match and load image etc...
dists = chi_square_statistics_fast(segment.shape_dist, model.all_shape_dists);
%dists = chi_square_statistics_fast(segment.edge_shape_dist, model.all_edge_shape_dists);
[~, idx] = sort(dists, 'ascend');
 
%% plotting the closest matches
num_to_plot = 20;
[p, q] = best_subplot_dims(num_to_plot);

subplot(p, q, 1)
imagesc2(boxcrop_2d(segment.mask))
set(gcf, 'color', [1,1, 1])

for ii = 1:(num_to_plot-1)
    
    this.model_idx = model.all_model_idx(idx(ii));
    this.model = params.model_filelist{this.model_idx};
    this.view = model.all_view_idx(idx(ii));
    this.path = sprintf(paths.basis_models.rendered, this.model, this.view);
    load(this.path, 'depth')
    
    subplot(p, q, ii+1)
    plot_depth(depth)
    title([num2str(this.model_idx) ' - ' num2str(this.view)])
    
end

%% idea now is to rotate each image to the best alignment.
profile on
[~, ~, segment.angle_hist] = edge_normals(segment.mask, 15);
plot(segment.angle_hist)
for jj = 1:length(segment.angle_hist)
    segment.angle_hists(jj, :) = circshift(segment.angle_hist(:), jj)';
end
all_angles = linspace(0, 360, length(segment.angle_hist));
profile off viewer
%% Rotate and plot the shapes
% To do - save all the edge normals offline and then load them in. THis
% would cause them to be quick
for ii = 1:(num_to_plot-1)
    
    this.model_idx = model.all_model_idx(idx(ii));
    this.model = params.model_filelist{this.model_idx};
    this.view = model.all_view_idx(idx(ii));
    this.path = sprintf(paths.basis_models.rendered, this.model, this.view);
    load(this.path, 'depth')
    depth = format_depth(depth);
    
    % finding the best rotation
    [~, ~, this.T] = edge_normals(~isnan(depth), 5);
    dists = chi_square_statistics_fast(this.T', segment.angle_hists);
    [~, dist_ind] = min(dists);
    
    this.angle = all_angles(dist_ind);
    
    % plotting the rotated depth image 
    subplot(p, q, ii+1)
    t_depth = imrotate(depth, -this.angle);
    t_depth(t_depth==0) = nan;
    t_depth = boxcrop_2d(t_depth);
    plot_depth(t_depth)
    title([num2str(this.model_idx) ' - ' num2str(this.view)])
    ii
end
profile off viewer

%% 
clf
iii = 1
plot(model.all_shape_dists(idx(iii), :));
hold on
plot(segment.shape_dist,'r')
hold off


%% plotting some random database shapes
addpath(genpath('../../common/'))
addpath ../../2D/src/utils

for ii = 1:25
    
    this_idx = randi(length(model.all_model_idx));
    this.model_idx = model.all_model_idx(this_idx);
    this.model = params.model_filelist{this.model_idx};
    this.view = model.all_view_idx(this_idx);
    this.path = sprintf(paths.basis_models.rendered, this.model, this.view);
    load(this.path, 'depth')
    max_depth = max(depth(:));
    depth(abs(depth-max_depth)<0.01) = 0;
    depth = boxcrop_2d(depth);
    depth(depth==0) =nan;
    subaxis(5, 5, ii, 'Margin',0, 'Spacing', 0)
    imagesc(depth)
    axis image off
    colormap(flipud(gray))    
end
set(gcf, 'color', [1, 1, 1])
