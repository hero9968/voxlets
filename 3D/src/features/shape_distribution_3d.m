function fv = shape_distribution_3d(XYZ, opts)
% compute shape distribution for 2d points

num_samples = opts.num_samples;
bin_edges = opts.bin_edges;
rescaling = opts.rescaling;

assert(size(XYZ, 2) == 3);

% extracting data from inputs
num_points = size(XYZ, 1);
X = XYZ(:, 1);
Y = XYZ(:, 2);
Z = XYZ(:, 3);

% choosing indices of random points
inds1 = randi(num_points, 1, num_samples);
inds2 = randi(num_points, 1, num_samples);

% computing the distances between the random pairs of points
dists = (X(inds1) - X(inds2)).^2 + (Y(inds1) - Y(inds2)).^2 + (Z(inds1) - Z(inds2)).^2;
dists = sqrt(dists);

if rescaling
    % renormalising dists by dividing through by the scale
    scale_factor = median(dists);
    dists = dists / scale_factor;
end

to_remove = dists < min(bin_edges) | dists > max(bin_edges) | dists == 0;
dists( to_remove ) = [];



%fraction_removed = sum(to_remove) / numel(to_remove);
%disp(['Removing ' num2str(fraction_removed)])
%disp(['Average length is ' num2str(mean(dists))])

[fv, ~] = histc(dists, bin_edges);
fv = fv/sum(fv);