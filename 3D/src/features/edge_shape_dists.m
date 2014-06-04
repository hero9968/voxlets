function [fv, dists] = edge_shape_dists(mask, dict)
% need to do some sort of rescaling...
% perhaps similar to in the 3D case?

[X, Y] = find(edge(mask));
num_points = length(X);

num_samples = 10000;

% choosing indices of random points
inds1 = randi(num_points, 1, num_samples);
inds2 = randi(num_points, 1, num_samples);

% computing the distances between the random pairs of points
dists = (X(inds1) - X(inds2)).^2 + (Y(inds1) - Y(inds2)).^2;
dists = sqrt(dists);

% normalising for scale
scale = prctile(dists, 95);
dists = dists / scale;

if nargin == 2
    [~, idx] = pdist2(dict, dists(:), 'Euclidean', 'Smallest', 1);
    fv = accumarray(idx(:), 1, [size(dict, 1), 1]);
    fv = fv(:)' / sum(fv);
else
    fv = [];
end
