function segment = extract_segment(cloud, idxs, params)
% helper function to extract the points belonging to a specified segment
% from the point cloud. Not a very clever way of doing this but useful in
% the short run.

segment.idx = idxs;
segment.mask = reshape(segment.idx, [480, 640]) > 0.1;
segment.xyz = cloud.xyz(segment.idx>0.5, :);
segment.norms = cloud.normals(segment.idx>0.5, :);
segment.scaled_xyz = segment.xyz * normalise_scale(segment.xyz);

% computing features for the segment...
segment.shape_dist = shape_distribution_norms_3d(segment.scaled_xyz, segment.norms, params.shape_dist);
segment.edge_shape_dist = edge_shape_dists_norms(segment.mask, params.shape_dist.edge_dict);
