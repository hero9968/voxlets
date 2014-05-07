% a script to load in a depth image, convert to xyz, compute normals and
% segment
clear
cd ~/projects/shape_sharing/3D/features/
addpath ../plotting/
addpath ../transformations/
addpath ../../2D/src/segment/
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
[cloud.normals, cloud.curve] = normals_wrapper(cloud.xyz, 'knn', 150);

%% plot normals
plot_normals(cloud.xyz, cloud.normals, 0.05)

%% now do some kind of segmentation...
opts.min_cluster_size = 50;
opts.max_cluster_size = 1e6;
opts.num_neighbours = 50;
opts.smoothness_threshold = (7.0 / 180.0) * pi;
opts.curvature_threshold = 1.0;

[idx] = segment_wrapper(cloud.xyz, cloud.normals, cloud.curve, opts);
nansum(idx)

imagesc(reshape(idx, 240, 320))

%% perhaps doing segmentation with different parameters and combining?
smoothness_thresholds = ([1:5:20] / 180 ) * pi;
curvature_thresholds = 0.3:0.2:1;

N = length(smoothness_thresholds) * length(curvature_thresholds);

all_idx = cell(1, N);
count = 1;

for ii = 1:length(smoothness_thresholds)
    for jj = 1:length(curvature_thresholds)
        
        opts.smoothness_threshold = smoothness_thresholds(ii);
        opts.curvature_threshold = curvature_thresholds(jj);
        
        [temp_idx] = segment_wrapper(cloud.xyz, cloud.normals, cloud.curve, opts);
        
        all_idx{count} = temp_idx;
        count = count + 1;
        
        length(unique(temp_idx))

    end
end

segmented_matrix = cell2mat(all_idx);
imagesc(single(segmented_matrix));

%% merging together identical indices, to form a bianry array of unique segmentations

% removing the nans before starting
nan_points = any(isnan(segmented_matrix), 2);
segmented_nans_removed = segmented_matrix;
segmented_nans_removed(nan_points, :) = [];

% first remove segmentations (NOT segments) which are exactly the same
size(segmented_nans_removed)
segmented_nans_removed = unique(segmented_nans_removed', 'rows')';
size(segmented_nans_removed)
imagesc(segmented_nans_removed)

%% now filtering
filter_opts.min_size = 100;
filter_opts.overlap_threshold = 0.7;

final_idx = filter_segments(segmented_nans_removed', filter_opts)';
imagesc(final_idx)

%% restoring the matrix
output_matrix = nan(size(segmented_matrix, 1), size(final_idx, 2));
output_matrix(~nan_points, :) = final_idx;

%% plotting
clf
output_matrix2 = output_matrix;%segments_to_binary(segmented_matrix')';

for ii = 1:size(output_matrix2, 2)
    
    temp_image = reshape(output_matrix2(:, ii), 240, []);
    
    subplot(4,4, ii)
    imagesc(temp_image);
    axis image
    
    
end

%% other tyoe  of plotting
clf
for ii = 1:length(all_idx)
    
    temp_image = reshape(all_idx{ii}, 240, []);
    temp_image(isnan(temp_idx)) = -1;
    
    subplot(5,4, ii)
    imagesc(temp_image);
    axis image
    
    
end

