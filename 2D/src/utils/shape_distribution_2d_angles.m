function [fv, dists_original, angles_original] = shape_distribution_2d_angles(XY, norms, num_samples, xy_bin_edges, angles_bin_edges, hist_2d)
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
dot_prod = dot(norms(:, inds1), norms(:, inds2), 1);
dot_prod = min(max(dot_prod, -1), 1);
angles = acos(dot_prod); % range is [0, pi];

% removing distances out of range
dists_original = dists;
angles_original = angles;
to_remove = dists < min(xy_bin_edges) | dists > max(xy_bin_edges) | dists == 0;

% choosing what type of histogram to make
if hist_2d
    
    % remove outliers from both dists and angles
    dists( to_remove ) = [];
    angles( to_remove ) = [];

    % forming the 2d histogram
    histogram = hist2(dists, angles, xy_bin_edges, angles_bin_edges);
    histogram = histogram(:);
    
else
    
    % only remove outliers from the dists (as treating each independently)
    dists( to_remove ) = [];

    % making two 1-d histograms
    [histogram_1, ~] = histc(dists, xy_bin_edges);
    [histogram_2, ~] = histc(angles, angles_bin_edges);

    % concatenating 
    histogram = [histogram_1, histogram_2];
    
end

% normalising histogram
fv = histogram / sum(histogram);
fv = fv(:)';