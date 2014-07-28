function [fv, original_dists, original_angles] = shape_distribution_norms_3d(XYZ, norms, opts)
% compute shape distribution for 2d points

num_samples = opts.num_samples;
dict = opts.dict;

assert(size(XYZ, 2) == 3);
assert(size(XYZ, 1) == size(norms, 1));

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
original_dists = dists;

% computing the angles
dot_prod = dot(norms(inds1, :), norms(inds2, :), 2);
dot_prod = min(max(dot_prod, -1), 1);
angles = acos(dot_prod); % range is [0, pi];
original_angles = angles;

% removing nans
to_remove = isnan(dists) | isnan(angles);
dists(to_remove) = [];
angles(to_remove) = [];

if isfield(opts, 'just_dists') && opts.just_dists
    
    fv = [];
    
else

    [~, idx] = pdist2(double(dict), [dists(:), angles(:)], 'Euclidean', 'Smallest', 1);
    fv = accumarray(idx(:), 1, [size(dict, 1), 1]);
    fv = fv(:)' / sum(fv);
    
end

%fraction_removed = sum(to_remove) / numel(to_remove);
%disp(['Removing ' num2str(fraction_removed)])
%disp(['Average length is ' num2str(mean(dists))])

%[fv, ~] = histc(dists, bin_edges);
%fv = fv/sum(fv);