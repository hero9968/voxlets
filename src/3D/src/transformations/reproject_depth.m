function [xyz, input_mask] = reproject_depth(depth, K, max_depth)
% function to project depth image into real world coordinates!

im_height = size(depth, 1);
im_width = size(depth, 2);

% stack of homogeneous coordinates of each image cell
[xgrid, ygrid] = meshgrid(1:im_width, 1:im_height);
full_stack = [xgrid(:) .* depth(:), ygrid(:).* depth(:), depth(:)];

% apply inverse intrinsics, and convert to standard coloum format
xyz = (K \ full_stack')';

% removing points at max depth
if nargin > 2
    if isnan(max_depth)
        to_remove = any(isnan(xyz), 2);
    else
        to_remove = abs(depth-max_depth) < 0.001;
    end
    xyz(to_remove, :) = [];
    input_mask = reshape(~to_remove, [im_height, im_width]);
end