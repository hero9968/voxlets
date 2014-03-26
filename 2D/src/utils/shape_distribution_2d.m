function fv = shape_distribution_2d(X, Y, num_samples, bin_edges)
% compute shape distribution for 2d points

num_points = length(X);

% choosing indices of random points
inds1 = randi(num_points, 1, num_samples);
inds2 = randi(num_points, 1, num_samples);

% computing the distances between the random pairs of points
dists = (X(inds1) - X(inds2)).^2 + (Y(inds1) - Y(inds2)).^2;
dists = sqrt(dists);

to_remove = dists < min(bin_edges) | dists > max(bin_edges);
dists( to_remove ) = [];

[fv, ~] = histc(dists, bin_edges);