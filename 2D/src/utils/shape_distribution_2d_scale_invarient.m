function fv = shape_distribution_2d_scale_invarient(X, Y, num_samples, bin_edges)
% compute shape distribution for 2d points

to_remove = isnan(X) | isnan(Y);
X(to_remove) = [];
Y(to_remove) = [];

num_points = length(X);

% choosing indices of random points
inds1 = randi(num_points, 1, num_samples);
inds2 = randi(num_points, 1, num_samples);

% computing the distances between the random pairs of points
dists = (X(inds1) - X(inds2)).^2 + (Y(inds1) - Y(inds2)).^2;
dists = sqrt(dists);

% now scaling all the distances by the median distance
scale_factor = median(dists) * 5;
dists = dists / scale_factor;

% remving points out of range of the bin edges
to_remove = dists < min(bin_edges) | dists > max(bin_edges);
if sum(to_remove) > 0.1 * length(dists)
    warning('Removing too many points...')
end
dists( to_remove ) = [];

% now forming the histogram
[fv, ~] = histc(dists, bin_edges);
fv = fv / sum(fv);
