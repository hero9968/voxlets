function xyz = depth2xyz( depth_in, max_depth )
% wrapper for depthToCloud

t_xyz = depthToCloud( depth_in );
t_xyz = permute(t_xyz, [3, 1, 2]);
xyz = t_xyz(:, :)';

% removing points at max depth
if nargin==2
    to_remove = abs(depth_in-max_depth) < 0.001;
    xyz(to_remove(:), :) = [];
end
