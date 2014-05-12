function fv = shape_distribution_3d(XYZ, num_samples, bin_edges)
% compute shape distribution for 2d points

assert(size(XYZ, 1) == 3);

% extracting data from inputs
num_points = size(XYZ, 2);
X = XYZ(1, :);
Y = XYZ(2, :);
Z = XYZ(3, :);

% choosing indices of random points
inds1 = randi(num_points, 1, num_samples);
inds2 = randi(num_points, 1, num_samples);

% computing the distances between the random pairs of points
dists = (X(inds1) - X(inds2)).^2 + (Y(inds1) - Y(inds2)).^2 + (Z(inds1) - Z(inds2)).^2;
dists = sqrt(dists);

to_remove = dists < min(bin_edges) | dists > max(bin_edges) | dists == 0;
dists( to_remove ) = [];

[fv, ~] = histc(dists, bin_edges);
fv = fv/sum(fv);