function [fv, original_dists] = shape_distribution_3d(XYZ, opts)
% compute shape distribution for 2d points

num_samples = opts.num_samples;
dict = opts.dict;

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
original_dists = dists;

[~, idx] = pdist2(dict, [dists(:)], 'Euclidean', 'Smallest', 1);

fv = accumarray(idx(:), 1, [size(dict, 1), 1]);
sum(idx==1)/length(idx);
%plot(fv)
%hold on
fv = fv(:)' / sum(fv);



%fraction_removed = sum(to_remove) / numel(to_remove);
%disp(['Removing ' num2str(fraction_removed)])
%disp(['Average length is ' num2str(mean(dists))])

%[fv, ~] = histc(dists, bin_edges);
%fv = fv/sum(fv);