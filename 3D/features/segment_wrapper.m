function [idx_out] = segment_wrapper( xyz, normals )
% matlab wrapper for fpfh mex function

% remove nan
n = size(xyz, 1);
to_remove = any(isnan(xyz), 2);
xyz(to_remove, :) = [];
normals(to_remove, :) = [];

% convert to double
xyz = double(xyz);
normals = double(normals);

% doing computation
temp_idx_out = segment_mex( xyz, normals );

% reforming idx
idx_out = nan(n, 1);
idx_out(~to_remove, :) = temp_idx_out;
