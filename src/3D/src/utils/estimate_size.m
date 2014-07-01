function scale = estimate_size(XYZ)
% finds a scale factor for points XY which will map them into some kind of
% consistant scale space
% let's do this so the distance between the furthest pair of points is 1
%
% could use e.g. 98th percentile instead of max to avoid problems with outliers

assert(size(XYZ, 2) == 3);

% taking two sets of random 1000 points
% extracting data from inputs
num_points = size(XYZ, 1);
X = XYZ(:, 1);
Y = XYZ(:, 2);
Z = XYZ(:, 3);

% choosing indices of random points
num_samples = 20000;
inds1 = randi(num_points, 1, num_samples);
inds2 = randi(num_points, 1, num_samples);

% computing the distances between the random pairs of points
dists = (X(inds1) - X(inds2)).^2 + (Y(inds1) - Y(inds2)).^2 + (Z(inds1) - Z(inds2)).^2;
dists = sqrt(dists);

%overall_scale = max(distances);
overall_scale = prctile(dists, 95);
scale = overall_scale;
%scale = 1 / overall_scale; 