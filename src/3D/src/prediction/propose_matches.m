function matches = propose_matches(segment, model, num_to_propose, feature_to_use, params, paths)
% uses the model to propose matches for the segment
% unsure exactly the form the matches might take, but could be related to
% some form of 
% Perhaps matches is a structure with the following form:
% matches(1).model
% matches(1).view
% matches(1).distance
if nargin < 4 || strcmp(feature_to_use, 'shape_dist')
    segment_features = segment.features.shape_dist;
    model_features = model.all_shape_dists;
elseif strcmp(feature_to_use, 'edge_shape_dist')
    segment_features = segment.features.edge_shape_dist;
    model_features = model.all_edge_shape_dists;
else
    error('Unknown feature vector') 
end

dists = chi_square_statistics_fast(segment_features, model_features);
[~, idx] = sort(dists, 'ascend');

for ii = 1:num_to_propose
    
    matches(ii).model.idx = model.all_model_idx(idx(ii));
    matches(ii).model.view = model.all_view_idx(idx(ii));
    matches(ii).chi_square.dist = dists(idx(ii));
    matches(ii).chi_square.position = ii;
    
    matches(ii).model.name = params.model_filelist{matches(ii).model.idx};
    matches(ii).model.path = sprintf(paths.basis_models.rendered, matches(ii).model.name, matches(ii).model.view);
    
    load(matches(ii).model.path, 'depth')
    depth = format_depth(depth);
    %[~, ~, this_T1] = edge_normals(~isnan(depth), 5);
    this_T = model.all_edge_angles_fv(idx(ii), :);

    angle_dists = chi_square_statistics_fast(this_T(:)', segment.features.angle_hists);
    [~, dist_ind] = min(angle_dists);
    matches(ii).transforms.angle = segment.features.all_angles(dist_ind);
    
    % ultimately this next bit will be taken offline...
    this.model = params.model_filelist{matches(ii).model.idx};
    this.path = sprintf(paths.basis_models.rendered, this.model, matches(ii).model.view);
    load(this.path, 'depth')
    matches(ii).transforms.centroid = centroid(~isnan(depth));
    matches(ii).depth = depth;
    matches(ii).transforms.median_depth = nanmedian(depth(:));    
    matches(ii).mask = ~isnan(depth);
    
    % getting the model scale ? this too will be taken offline in the future
    t_xyz = reproject_depth(matches(ii).depth, params.half_intrinsics);
    matches(ii).transforms.scale = estimate_size(t_xyz);
    matches(ii).xyz = t_xyz(matches(ii).mask(:), :);%t_xyz(matches(ii).mask(:), :) / matches(ii).scale;
    
    % getting the 3d centroid of the rendered image ? also can take this offline!
    temp_mask = +matches(ii).mask;
    temp_mask(matches(ii).mask) = 1:sum(sum(matches(ii).mask));
    
    % want centroid linear index to be cloest position in the mask to the
    % point... maybe the point is in a hole on the mask!
    [XX, YY] = find(temp_mask);
    [~, neighbour_idx] = pdist2([YY, XX], matches(ii).transforms.centroid, 'euclidean', 'smallest', 1);
    matches(ii).transforms.centroid_on_mask = [XX(neighbour_idx), YY(neighbour_idx)];
    linear_index = temp_mask(XX(neighbour_idx), YY(neighbour_idx));
    %matches(ii).centroid_3d = matches(ii).xyz(linear_index, :);
%    warning('Median')
    matches(ii).transforms.centroid_3d = nanmedian(matches(ii).xyz, 1);
    
    %imagesc(matches(ii).mask);
    %hold on
    %plot(YY(idx), XX(idx), 'r+', 'markersize', 10)
    %hold off
    %matches(ii).centroid_linear_index = temp_mask(round(matches(ii).centroid(2)), round(matches(ii).centroid(1)));
    %matches(ii).centroid_3d = matches(ii).xyz(matches(ii).centroid_linear_index, :);
    %break
    % consider taking a patch around the centroid and aligning the normals
    % of the two patches to get the normal alignment? Similar to Drost?
    
    [~, neighbour_idx] = pdist2(matches(ii).xyz, matches(ii).transforms.centroid_3d, 'euclidean', 'smallest', 1000);
    neighbour_xyz = matches(ii).xyz(neighbour_idx, :);
    matches(ii).transforms.centroid_normal = calcNormal( neighbour_xyz, matches(ii).transforms.centroid_3d);

end
