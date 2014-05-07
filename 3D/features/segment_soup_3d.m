function [output_matrix, final_idx] = segment_soup_3d( cloud, opts )
% forms a segmentation of 3D data into multiple different segmentations
% cloud is a structure with xyz, normals, curvature as fields
% opts is a structure with options

% setting parameters for segmentation and combination
smoothness_thresholds = ((1:5:20) / 180 ) * pi;
curvature_thresholds = 1;

filter_opts.min_size = opts.min_cluster_size;
filter_opts.overlap_threshold = opts.overlap_threshold;

% removing nans from the cloud
[cloud_filtered, idxs_removed] = remove_nans(cloud);
        
% doing segmentation with different parameters
N = length(smoothness_thresholds) * length(curvature_thresholds);

all_idx = cell(1, N);
count = 1;

for ii = 1:length(smoothness_thresholds)
    for jj = 1:length(curvature_thresholds)
        
        opts.smoothness_threshold = smoothness_thresholds(ii);
        opts.curvature_threshold = curvature_thresholds(jj);
        
        [all_idx{count}] = segment_wrapper(cloud_filtered, opts);
        count = count + 1;
        
        disp(['Done ' num2str(count) ' of ' num2str(N)])
    end
end

% combining all the segmentations together
segmented_matrix = cell2mat(all_idx);

% merging together identical indices, to form a binary array of unique segmentations

% first remove segmentations (NOT segments) which are exactly the same
segmented_matrix = unique(segmented_matrix', 'rows')';

% now filtering and converting to binary
% (perhaps would make more sense to convert to binary here, then filter)
final_idx = filter_segments(segmented_matrix', filter_opts)';

% restoring the matrix to its full size
output_matrix = zeros(size(cloud.xyz, 1), size(final_idx, 2));
output_matrix(~idxs_removed, :) = final_idx;




function [cloud_filtered, idxs_removed] = remove_nans(cloud)
% removes points with nans from the xyz, normals and curvature fields

idxs_removed = any(isnan(cloud.xyz), 2) | ...
            any(isnan(cloud.normals), 2) | ...
            isnan(cloud.curvature);
        
cloud_filtered = cloud;

cloud_filtered.xyz(idxs_removed, :) = [];
cloud_filtered.normals(idxs_removed, :) = [];
cloud_filtered.curvature(idxs_removed) = [];



