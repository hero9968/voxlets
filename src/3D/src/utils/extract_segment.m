function segment = extract_segment(cloud, idxs, params)
% helper function to extract the points belonging to a specified segment
% from the point cloud. Not a very clever way of doing this but useful in
% the short run.

segment.cloud.idx = idxs;
if numel(idxs) == 480*640
    segment.cloud.mask = reshape(segment.cloud.idx, [480, 640]) > 0.5;
elseif numel(idxs) == 240*320
    segment.cloud.mask = reshape(segment.cloud.idx, [240, 320]) > 0.5;
end
segment.cloud.xyz = cloud.xyz(segment.cloud.idx>0.5, :);
segment.cloud.norms = cloud.normals(segment.cloud.idx>0.5, :);
segment.cloud.scale = estimate_size(segment.cloud.xyz);
segment.cloud.scaled_xyz = segment.cloud.xyz / segment.cloud.scale;

% computing the features for the segment
segment.features = compute_segment_features(segment.cloud, params);

segment.transforms = compute_segment_transforms(segment.cloud);

segment.transforms.rotate_to_plane = cloud.segment.rotate_to_plane;

