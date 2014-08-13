function transforms = compute_segment_transforms(segment_cloud, base_cloud, params)
% computes transforms for the segment
% cloud is a structure with the fields:
%  - scaled_xyz, xyz points scaled to a suitable scale
%  - norms, a matrix of normals the same size as scaled_xyz
%  - mask, a logical array showing where the segment points came from in
%           original 2d image

transforms.centroid = centroid(segment_cloud.mask);
%transforms.median_depth = median(cloud.depth(segment_cloud.idx==1));

centroid_linear_index = ...
    centroid_2d_to_linear_index(transforms.centroid, segment_cloud.mask);

transforms.centroid_3d.xyz = nanmedian(segment_cloud.xyz, 1);
transforms.centroid_3d.norm = segment_cloud.normals(centroid_linear_index, :);

% compute the centroid normal - not just from the point but from a few more also...
[~, neighbour_idx] = ...
    pdist2(segment_cloud.xyz, transforms.centroid_3d.xyz, 'euclidean', 'smallest', 1000);

neighbour_xyz = segment_cloud.xyz(neighbour_idx, :);
transforms.centroid_normal = calcNormal( neighbour_xyz, transforms.centroid_3d.xyz );

% FINAL TRANSFORMATION MATRIX

% translation from the origin to the scene segment
trans2 = translation_matrix_3d(transforms.centroid_3d.xyz);
rot2 = inv(transformation_matrix_from_vector(transforms.centroid_normal, 1));
scale_segment = scale_matrix_3d(segment_cloud.scale);

% combining
transforms.final_M = trans2 * rot2 * scale_segment; 

%transforms.rotate_to_plane = cloud.segment.rotate_to_plane;
