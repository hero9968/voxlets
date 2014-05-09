function xyz = reproject_depth(depth, K, max_depth)
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
    to_remove = abs(depth-max_depth) < 0.001;
    xyz(to_remove, :) = [];
end
