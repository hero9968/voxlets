function xyz = reproject_depth(depth, K, top_left)
% function to project depth image into real world coordinates!

im_height = size(depth, 1);
im_width = size(depth, 2);

if nargin == 2
    top_left = [1, 1];
end

% stack of homogeneous coordinates of each image cell
u = top_left(1):(top_left(1)+im_width-1);
v = top_left(2):(top_left(2)+im_height-1);
[xgrid, ygrid] = meshgrid(u, v);
full_stack = [xgrid(:) .* depth(:), ygrid(:).* depth(:), depth(:)];

% apply inverse intrinsics, and convert to standard coloum format
xyz = (K \ full_stack')';
