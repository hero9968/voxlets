function features = compute_segment_features(cloud, params)
% computes features for the segment
% cloud is a structure with the fields:
%  - scaled_xyz, xyz points scaled to a suitable scale
%  - norms, a matrix of normals the same size as scaled_xyz
%  - mask, a logical array showing where the segment points came from in
%           original 2d image

features.shape_dist = ...
    shape_distribution_norms_3d(cloud.scaled_xyz, cloud.norms, params.shape_dist);

features.edge_shape_dist = ...
    edge_shape_dists_norms(cloud.mask, params.shape_dist.edge_dict);

% computing the feature vector fot eh sillhouette of the object
%[~, ~, segment.angle_hist] = edge_normals(segment.mask, 15);
features.angle_hist = edge_angle_fv(cloud.mask, 50);
%plot(segment.angle_hist)

for jj = 1:length(features.angle_hist)
    features.angle_hists(jj, :) = circshift(features.angle_hist(:), jj)';
end

features.all_angles = linspace(0, 360, length(features.angle_hist));
