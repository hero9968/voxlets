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

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
warning('Only using one model -view')
model_view_to_use = 6;
model_to_use = 4;
to_remove =  model.all_model_idx ~= model_to_use | model.all_view_idx ~= model_view_to_use;
%model_features(to_remove, :)  = inf;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

dists = chi_square_statistics_fast(segment_features, model_features);
[~, idx] = sort(dists, 'ascend');

for ii = 1:num_to_propose
    
    % extracting the basic information about this view being matched
    matches(ii).model.idx = model.all_model_idx(idx(ii));
    matches(ii).model.view = model.all_view_idx(idx(ii));
    matches(ii).model.name = params.model_filelist{matches(ii).model.idx};
    matches(ii).model.path = sprintf(paths.basis_models.rendered, matches(ii).model.name, matches(ii).model.view);
    
    matches(ii).chi_square.dist = dists(idx(ii));
    matches(ii).chi_square.position = ii;
    
    % finding the in-plane camera rotation
    load(matches(ii).model.path, 'depth')
    matches(ii).depth = format_depth(depth);
    this_T = model.all_edge_angles_fv(idx(ii), :);
    
    angle_dists = chi_square_statistics_fast(this_T(:)', segment.features.angle_hists);
    [~, dist_ind] = min(angle_dists);
    matches(ii).transforms.angle = segment.features.all_angles(dist_ind);
    
    % finding the transformations to the matched region
    % (ultimately this next bit will be taken offline...)
    matches(ii).transforms.centroid = centroid(~isnan(matches(ii).depth));
    matches(ii).transforms.median_depth = nanmedian(matches(ii).depth(:));    
    matches(ii).mask = ~isnan(matches(ii).depth);
    
    % getting the model scale ? this too will be taken offline in the future
    t_xyz = reproject_depth(matches(ii).depth, params.half_intrinsics);
    t_xyz(:, 2:3) = -t_xyz(:, 2:3);
    matches(ii).transforms.scale = estimate_size(t_xyz);
    matches(ii).xyz = t_xyz(matches(ii).mask(:), :);%t_xyz(matches(ii).mask(:), :) / matches(ii).scale;
    
    % here will transform points to be in OBJECT coordinates - also can take this offline!
    rot_name = sprintf('/Users/Michael/projects/shape_sharing/data/3D/basis_models/halo/mat_%d.csv', matches(ii).model.view);
    T = csvread(rot_name);
    matches(ii).transforms.inplane_rotation = in_camera_rotation_correction(T);
    matches(ii).transforms.render_projection = T;
    
    % NOTE - would also have to do norms here if retaining them
    
    % getting the 3d centroid of the rendered image - also can take this offline!
    temp_mask = +matches(ii).mask;
    temp_mask(matches(ii).mask) = 1:sum(sum(matches(ii).mask));
    
    % want centroid linear index to be cloest position in the mask to the
    % point... maybe the point is in a hole on the mask!
    %{
        [XX, YY] = find(temp_mask);
        [~, neighbour_idx] = pdist2([YY, XX], matches(ii).transforms.centroid, 'euclidean', 'smallest', 1);
        matches(ii).transforms.centroid_on_mask = [XX(neighbour_idx), YY(neighbour_idx)];
        linear_index = temp_mask(XX(neighbour_idx), YY(neighbour_idx));
        %matches(ii).centroid_3d = matches(ii).xyz(linear_index, :);
    %    warning('Median')
    %}
    
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
    
    % loading in the voxel data
    voxel_filename = sprintf('/Users/Michael/projects/shape_sharing/data/3D/basis_models/voxelised/%s.mat', matches(ii).model.name);
    vox_struct = load(voxel_filename);
    V = vox_struct.vol;
    V(V<30) = 0;
    V = permute(V, [2, 1, 3]);
    [inds] = find(V);
    [i, j, k] = ind2sub(size(V), inds);
    trans_vox = apply_transformation_3d([i,j,k], params.voxelisation.T_vox);
    %matches(ii).vox_xyz = trans_vox;
    matches(ii).vox_xyz = [i, j, k];%apply_transformation_3d(trans_vox, inv(T));
    matches(ii).transforms.vox_inv = inv(T);
    
    [~, neighbour_idx] = pdist2(matches(ii).xyz, matches(ii).transforms.centroid_3d, 'euclidean', 'smallest', 1000);
    neighbour_xyz = matches(ii).xyz(neighbour_idx, :);
    matches(ii).transforms.centroid_normal = calcNormal( neighbour_xyz, matches(ii).transforms.centroid_3d);

    % FINAL TRANSFORMATION MATRICES
    
    % translating the basis shape to the origin
    trans1 = translation_matrix_3d(-matches(ii).transforms.centroid_3d);
    rot1 = (transformation_matrix_from_vector(matches(ii).transforms.centroid_normal, 1));
    
    % resolving the rotation in the camera plane
    camera_rot = rotation_matrix(matches(ii).transforms.angle + 1.5*7.2);% - matches(ii).transforms.inplane_rotation);
    camera_rot = [1, 0, 0, 0; zeros(3, 1), camera_rot];
    
    scale_matches = scale_matrix_3d(1 / matches(ii).transforms.scale);
    
    matches(ii).transforms.final_M = scale_matches * camera_rot * rot1 * trans1;
    
end
