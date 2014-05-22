function [fv, dists_original] = shape_distribution_2d(XY, num_samples, bin_edges)
% compute shape distribution for 2d points

assert(size(XY, 1) == 2);

% extracting data from inputs
num_points = size(XY, 2);
X = XY(1, :);
Y = XY(2, :);

% choosing indices of random points
inds1 = randi(num_points, 1, num_samples);
inds2 = randi(num_points, 1, num_samples);

% computing the distances between the random pairs of points
dists = (X(inds1) - X(inds2)).^2 + (Y(inds1) - Y(inds2)).^2;
dists = sqrt(dists);
dists_original = dists;

to_remove = dists < min(bin_edges) | dists > max(bin_edges) | dists == 0;
dists( to_remove ) = [];

[fv, ~] = histc(dists, bin_edges);
fv = fv/sum(fv);