function scale = normalise_scale(XY)
% finds a scale factor for points XY which will map them into some kind of
% consistant scale space
% let's do this so the distance between the furthest pair of points is 1
%
% could use e.g. 98th percentile instead of max to avoid problems with outliers

assert(size(XY, 1) == 2);

if size(XY, 2) > 10000
    error('Probably too big for this method - think about a more sensible way');
end

distances = pdist(XY');

%overall_scale = max(distances);
overall_scale = prctile(distances, 95);

scale = 1 / overall_scale; 