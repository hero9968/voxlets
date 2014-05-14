function XY = xy_from_depth(depth)
% converts a rendered depth map into 2D points

assert(size(depth, 1) == 1)

% forming XY points
X = 1:length(depth);
Y = depth;
XY = [X; Y];