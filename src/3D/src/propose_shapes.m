% script to propose 3D basis shapes for a specific region, and to somehow
% visualise. No transformations or anything clever like that yet...

% a script to load in a depth image, convert to xyz, compute normals and segment
clear
cd ~/projects/shape_sharing/src/3D/src/
addpath(genpath('.'))
addpath(genpath('../../common/'))
addpath(genpath('../../2D/src'))
define_params_3d
load(paths.structured_model_file, 'model')

%% loading in some of the ECCV dataset
clear cloud
%filepath = '/Users/Michael/data/others_data/ECCV_dataset/pcd_files/frame_20111220T111153.549117.pcd';
%P = loadpcd(filepath);
%cloud.xyz = P(:, :, 1:3);
%cloud.xyz = reshape(permute(cloud.xyz, [3, 1, 2]), 3, [])';
P = loadpcd('~/projects/shape_sharing/data/3D/scenes/ren2_noisy00000.pcd');
%%
D = flipdim(permute(reshape(P, [6, 640, 480]), [1, 3, 2]), 2);
cloud.xyz = D(1:3, :)';
nan_locations = D(1, :)==0;
cloud.xyz(nan_locations(:), :) = nan;
cloud.depth = -squeeze(D(3, :, :));
cloud.depth(nan_locations) = nan;
[cloud.normals, cloud.curvature] = normals_wrapper(cloud.xyz, 'knn', 50);
cloud.rgb = repmat(cloud.depth, [1, 1, 3])/2;
clear P D

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
 
% plotting the closest matches
num_to_plot = 20;
[p, q] = best_subplot_dims(num_to_plot);

subplot(p, q, 1)
imagesc(boxcrop_2d(segment.mask))
axis image

for ii = 1:(num_to_plot-1)
    
    this.model_idx = model.all_model_idx(idx(ii));
    this.model = params.model_filelist{this.model_idx};
    this.view = model.all_view_idx(idx(ii));
    this.path = sprintf(paths.basis_models.rendered, this.model, this.view);
    load(this.path, 'depth')
    max_depth = max(depth(:));
    depth(abs(depth-max_depth)<0.01) = nan;
    
    subplot(p, q, ii+1)
    imagesc(depth)
    title([num2str(this.model_idx) ' - ' num2str(this.view)])
    axis image off
    colormap(flipud(gray))
    
end

%% idea now is to rotate each image to the best alignment.

[~, ~, segment.angle_hist] = edge_normals(segment.mask, 15);
plot(segment.angle_hist)
for jj = 1:length(segment.angle_hist)
    segment.angle_hists(jj, :) = circshift(segment.angle_hist(:), jj)';
end
all_angles = linspace(0, 360, length(segment.angle_hist));

%%
profile on
for ii = 1:(num_to_plot-1)
    
    this.model_idx = model.all_model_idx(idx(ii));
    this.model = params.model_filelist{this.model_idx};
    this.view = model.all_view_idx(idx(ii));
    this.path = sprintf(paths.basis_models.rendered, this.model, this.view);
    load(this.path, 'depth')
    max_depth = max(depth(:));
    depth(abs(depth-max_depth)<0.01) = nan;
    
    % finding the best rotation
    [~, ~, this.T] = edge_normals(~isnan(depth), 15);
    dists = chi_square_statistics_fast(this.T', segment.angle_hists);
    [~, dist_ind] = min(dists);
    
    this.angle = all_angles(dist_ind);
    
    
    subplot(p, q, ii+1)
    t_depth = imrotate(depth, -this.angle);
    t_depth(t_depth==0) = nan;
    t_depth = boxcrop_2d(t_depth);
    imagesc(t_depth)
    title([num2str(this.model_idx) ' - ' num2str(this.view)])
    axis image off
    colormap(flipud(gray))    
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
