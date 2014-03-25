function grid_out = fill_grid_from_depth(depth_in, image_height, fillvalue)
% the opposite of raytrace_2d.
% given a depth, will fill a grid to the specified max depth.
% will fill in the pixels hit with 1s, and the others with whatever is
% specified in the third argument

if nargin < 3
    fillvalue = 0.5;
end

width = length(depth_in);

% fill out grid
grid_out = zeros(image_height, width);

for ii = 1:width
    if depth_in(ii) > 0
        grid_out(depth_in(ii), ii) = 1;
        grid_out(depth_in(ii)+1:end, ii) = fillvalue;
    end
end

