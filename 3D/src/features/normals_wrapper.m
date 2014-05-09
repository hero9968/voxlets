function [norms_out, curve_out] = normals_wrapper( xyz, type, para )
% matlab wrapper for fpfh mex function

% remove nan
n = size(xyz, 1);
idx = any(isnan(xyz), 2);
xyz(idx, :) = [];
xyz = double(xyz);

% doing computation
[temp_norms_out, temp_curve_out] = normals_mex( xyz, type, para );

% reforming normals and curvature
norms_out = nan(n, 3);
norms_out(~idx, :) = temp_norms_out;

curve_out = nan(n, 1);
curve_out(~idx) = temp_curve_out;
