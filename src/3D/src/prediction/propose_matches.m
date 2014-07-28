function matches = propose_matches(segment, model, num_to_propose, feature_to_use, params, paths)
% uses the model to propose matches for the segment
% matches is a structure with the following form:
% matches(1).model
% matches(1).view
% matches(1).distance
% etc.

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
%warning('Only using one model -view')
model_view_to_use = 6;
model_to_use = 4;
to_remove =  model.all_model_idx ~= model_to_use | model.all_view_idx ~= model_view_to_use;
%model_features(to_remove, :)  = inf;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% apply the machine learning to propose shapes (here is just NN classification)
dists = chi_square_statistics_fast(segment_features, model_features);
[~, idx] = sort(dists, 'ascend');

for ii = 1:num_to_propose
    
    % BASIC INFORMATION ABOUT MATCH
    
    matches(ii).model.idx = model.all_model_idx(idx(ii));
    matches(ii).model.view = model.all_view_idx(idx(ii));
    matches(ii).model.name = params.model_filelist{matches(ii).model.idx};
    matches(ii).model.path = sprintf(paths.basis_models.rendered, matches(ii).model.name, matches(ii).model.view);
    
    matches(ii).chi_square.dist = dists(idx(ii));
    matches(ii).chi_square.position = ii;
       
    % INDIVIDUAL PARTS OF THE TRANSFORMATION
    
    % in-plane camera rotation
    load(matches(ii).model.path, 'depth')
    matches(ii).depth = format_depth(depth);
    this_T = model.all_edge_angles_fv(idx(ii), :);
    
    angle_dists = chi_square_statistics_fast(this_T(:)', segment.features.angle_hists);
    [~, dist_ind] = min(angle_dists);
    matches(ii).transforms.angle = segment.features.all_angles(dist_ind);
            
    % 2D match data
    %matches(ii).transforms.centroid = centroid(~isnan(matches(ii).depth));
    %matches(ii).transforms.median_depth = nanmedian(matches(ii).depth(:));    
    matches(ii).mask = ~isnan(matches(ii).depth);
    
    % scale
    t_xyz = reproject_depth(matches(ii).depth, params.half_intrinsics);
    t_xyz(:, 2:3) = -t_xyz(:, 2:3);
    matches(ii).transforms.scale = estimate_size(t_xyz);
    matches(ii).xyz = t_xyz(matches(ii).mask(:), :);   
        
    % 3d centroid
    matches(ii).transforms.centroid_3d = nanmedian(matches(ii).xyz, 1);
    
    % centroid normal (but from a wider area than the saved normal)
    [~, neighbour_idx] = pdist2(matches(ii).xyz, ...
        matches(ii).transforms.centroid_3d, 'euclidean', 'smallest', 1000);
    neighbour_xyz = matches(ii).xyz(neighbour_idx, :);
    matches(ii).transforms.centroid_normal = ...
        calcNormal( neighbour_xyz, matches(ii).transforms.centroid_3d);

    % transformation from the camera to the object centre
    rot_name = sprintf('/Users/Michael/projects/shape_sharing/data/3D/basis_models/halo/mat_%d.csv', matches(ii).model.view);
    matches(ii).transforms.vox_inv = inv(csvread(rot_name));
        
    % FINAL TRANSFORMATION MATRICES
    
    % translating the basis shape to the origin
    scale_matches = scale_matrix_3d(1 / matches(ii).transforms.scale);
    trans1 = translation_matrix_3d(-matches(ii).transforms.centroid_3d);
    rot1 = transformation_matrix_from_vector(matches(ii).transforms.centroid_normal, 1);
    camera_rot = rotation_matrix(matches(ii).transforms.angle + 1.5*7.2);
    camera_rot = [1, 0, 0, 0; zeros(3, 1), camera_rot];
    
    matches(ii).transforms.final_M = scale_matches * camera_rot * rot1 * trans1;
    
    % VOXEL DATA
    
    matches(ii).vox_xyz = load_vox(matches(ii).model.name);
        
    
end
