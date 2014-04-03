function fv = shape_distribution_2d_angles(XY, norms, num_samples, xy_bin_edges, angles_bin_edges)
% compute shape distribution for 2d points

% for now should I compute the normals in this function?


% input checks and setup
num_points = size(XY, 2);
assert(size(norms, 2) == num_points);
assert(size(norms, 1)==2 && size(XY, 1) == 2);

normal_lengths = sqrt(norms(1, :).^2+norms(2, :));
assert(all(abs(normal_lengths - 1) < 0.001), 'normals apparently not normalised');

X = XY(1, :);   
Y = XY(2, :);

% choosing indices of random points
inds1 = randi(num_points, 1, num_samples);
inds2 = randi(num_points, 1, num_samples);

% computing the distances between the random pairs of points
dists = (X(inds1) - X(inds2)).^2 + (Y(inds1) - Y(inds2)).^2;
dists = sqrt(dists);

% computing angles between the same random pairs
angles = dot(norms(:, inds1), norms(:, inds2), 2);% range is [-1, 1];

% removing distances out of range
to_remove = dists < min(bin_edges) | dists > max(bin_edges);
dists( to_remove ) = [];
angles( to_remove ) = [];

% forming the 2d histogram
histogram = hist2(dists, angles, xy_bin_edges, angles_bin_edges);
histogram = histogram(:);
fv = histogram / sum(histogram);
