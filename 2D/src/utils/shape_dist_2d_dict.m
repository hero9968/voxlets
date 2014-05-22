function [fv, dists, angles] = shape_dist_2d_dict(XY, norms, num_samples, dict)
% compute shape distribution for 2d points

% input checks and setup
num_points = size(XY, 2);
assert(size(norms, 2) == num_points);
assert(size(norms, 1)==2 && size(XY, 1) == 2);

normal_lengths = sqrt(norms(1, :).^2+norms(2, :).^2);
assert(all(abs(normal_lengths - 1) < 0.001), 'normals apparently not normalised');

% choosing indices of random points
inds1 = randi(num_points, 1, num_samples);
inds2 = randi(num_points, 1, num_samples);

% computing the distances between the random pairs of points
X = XY(1, :);   
Y = XY(2, :);
dists = (X(inds1) - X(inds2)).^2 + (Y(inds1) - Y(inds2)).^2;
dists = sqrt(dists);

% computing angles between the same random pairs
angles = dot(norms(:, inds1), norms(:, inds2), 1);% range is [-1, 1];

% finding nearest neighbours in the dictioanry
%idx = knnsearch(dict, [dists', angles']);
[~, idx] = pdist2(dict, [dists', angles'], 'Euclidean', 'Smallest', 1);

% computing the output array
fv = accumarray(idx(:), 1, [size(dict, 1), 1]);
fv = fv(:)' / sum(fv);