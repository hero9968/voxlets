function [idx_out, temp_idx_out] = segment_wrapper( xyz, normals, curve, opts )
% matlab wrapper for fpfh mex function

% remove nan
n = size(xyz, 1);
to_remove = any(isnan(xyz), 2) | any(isnan(normals), 2) | isnan(curve);
xyz(to_remove, :) = [];
normals(to_remove, :) = [];
curve(to_remove) = [];

% doing computation
temp_idx_out = segment_mex( double(xyz), double(normals), double(curve), ...
    int32(opts.min_cluster_size), int32(opts.max_cluster_size), ...
    int32(opts.num_neighbours), opts.smoothness_threshold, ...
    opts.curvature_threshold );

% reforming idx
idx_out = nan(n, 1);
idx_out(~to_remove, :) = temp_idx_out;
