% script to propose 3D basis shapes for a specific region, and to somehow
% visualise. No transformations or anything clever like that yet...

% a script to load in a depth image, convert to xyz, compute normals and segment
clear
cd ~/projects/shape_sharing/3D/src/
addpath plotting/
addpath features/
addpath ./file_io/matpcl/
addpath(genpath('../../common/'))
addpath transformations/
addpath utils/
addpath ../../2D/src/segment/
run ../define_params_3d.m
load(paths.structured_model_file, 'model')

%% loading in some of the ECCV dataset
clear cloud
filepath = '/Users/Michael/data/others_data/ECCV_dataset/pcd_files/frame_20111220T111153.549117.pcd';
P = loadpcd(filepath);
cloud.xyz = P(:, :, 1:3);
cloud.xyz = reshape(permute(cloud.xyz, [3, 1, 2]), 3, [])';
cloud.rgb = P(:, :, 4:6);
cloud.depth = reshape(P(:, :, 3), [480, 640]);
[cloud.normals, cloud.curvature] = normals_wrapper(cloud.xyz, 'knn', 50);

%% running segment soup algorithm
[idxs, idxs_without_nans, probabilities, all_idx] = segment_soup_3d(cloud, params.segment_soup);

%% plotting segments
close all
plot_segment_soup_3d(cloud.rgb.^0.2, idxs);
for ii = 1:length(probabilities)
    subplot(2, 4, ii)
    title(num2str(probabilities(ii)))
end
set(findall(gcf,'type','text'),'fontSize',18,'fontWeight','bold')

%% Choosing a segment and computing the feature vector
segment.seg_index = 4;
segment.idx = idxs(:, segment.seg_index);
segment.xyz = cloud.xyz(segment.idx>0.5, :);
segment.scaled_xyz = segment.xyz * normalise_scale(segment.xyz);
segment.shape_dist = shape_distribution_3d(segment.scaled_xyz, params.shape_dist);

%% Find closest match and load image etc...
dists = chi_square_statistics(segment.shape_dist, model.all_shape_dists);
%dists = kullback_leibler_divergence(segment.shape_dist, model.all_shape_dists);
[~, idx] = sort(dists, 'ascend');

%% plotting the closest matches
num_to_plot = 20;
[p, q] = best_subplot_dims(num_to_plot);

for ii = 1:num_to_plot
    
    this.model_idx = model.all_model_idx(idx(ii));
    this.model = params.model_filelist{this.model_idx};
    this.view = model.all_view_idx(idx(ii));
    this.path = sprintf(paths.basis_models.rendered, this.model, this.view);
    load(this.path, 'depth')
    depth(abs(depth-3)<0.01) = nan;
    
    subplot(p, q, ii)
    imagesc(depth)
    title([num2str(this.model_idx) ' - ' num2str(this.view)])
    axis image off
    colormap(flipud(gray))
    
    
end





