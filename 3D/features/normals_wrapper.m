function [norms_out, temp_norms_out] = normals_wrapper( xyz, type, para )
% matlab wrapper for fpfh mex function

% remove nan
n = size(xyz, 1);
idx = any(isnan(xyz), 2);
xyz(idx, :) = [];
xyz = double(xyz);

% doing computation
temp_norms_out = normals_mex( xyz, type, para );

% reforming normals
norms_out = nan(n, 3);
norms_out(~idx, :) = temp_norms_out;
