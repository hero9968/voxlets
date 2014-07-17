function segment = extract_segment(cloud, idxs, params)
% helper function to extract the points belonging to a specified segment
% from the point cloud. Not a very clever way of doing this but useful in
% the short run.

segment.cloud.idx = idxs;
segment.cloud.mask = reshape(segment.cloud.idx, [480, 640]) > 0.5;
segment.cloud.xyz = cloud.xyz(segment.cloud.idx>0.5, :);
segment.cloud.norms = cloud.normals(segment.cloud.idx>0.5, :);
segment.transforms.scale = estimate_size(segment.cloud.xyz);
segment.cloud.scaled_xyz = segment.cloud.xyz / segment.transforms.scale;

% computing features for the segment...
segment.features.shape_dist = ...
    shape_distribution_norms_3d(segment.cloud.scaled_xyz, segment.cloud.norms, params.shape_dist);
segment.features.edge_shape_dist = ...
    edge_shape_dists_norms(segment.cloud.mask, params.shape_dist.edge_dict);

% computing the feature vector fot eh sillhouette of the object
%[~, ~, segment.angle_hist] = edge_normals(segment.mask, 15);
segment.features.angle_hist = edge_angle_fv(segment.cloud.mask, 50);
%plot(segment.angle_hist)
for jj = 1:length(segment.features.angle_hist)
    segment.features.angle_hists(jj, :) = circshift(segment.features.angle_hist(:), jj)';
end
segment.features.all_angles = linspace(0, 360, length(segment.features.angle_hist));

segment.transforms.centroid = centroid(segment.cloud.mask);
segment.transforms.median_depth = median(cloud.depth(segment.cloud.idx==1));

centroid_linear_index = ...
    centroid_2d_to_linear_index(segment.transforms.centroid, segment.cloud.mask);
%centroid_linear_index
%[~, neighbour_idx] = pdist2(segment.xyz, segment.centroid_3d, 'euclidean', 'smallest', 1000);

%segment.centroid_3d.xyz = segment.xyz(centroid_linear_index, :);
segment.transforms.centroid_3d.xyz = nanmedian(segment.cloud.xyz, 1);
segment.transforms.centroid_3d.norm = segment.cloud.norms(centroid_linear_index, :);

% compute the centroid normal - not just from the point but from a few more
% also...
[~, neighbour_idx] = ...
    pdist2(segment.cloud.xyz, segment.transforms.centroid_3d.xyz, 'euclidean', 'smallest', 1000);

neighbour_xyz = segment.cloud.xyz(neighbour_idx, :);
segment.transforms.centroid_normal = calcNormal( neighbour_xyz, segment.transforms.centroid_3d.xyz );


