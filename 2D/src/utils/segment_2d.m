function idxs = segment_2d(depth, threshold, nms_width)
% simple function to segment the depth image defined by depth into mutliple
% regions, based a simple distance threshold on the first derivative. 
% Nothing clever going on here!
%
% INPUTS
%  depth - 1d depth image
%  threshold - threshold on allowing two points to join up
%  nms_width - width of non maximal suppresion window
%

% input checks
assert(isvector(depth))
assert(isscalar(threshold))
assert(isscalar(nms_width))

% generating the first derivative and doing NMS
first_deriv = diff(depth);
derv = abs([first_deriv, 0]);
derv_sup = non_max_sup_1d(derv, nms_width, 0)';

% finding the split points and shifting by one
split_points = derv_sup >= threshold;
split_points = circshift(split_points, [0, 1]);

% generating final labelling
idxs = cumsum(split_points);

% generating the heirachical linkage between the clusters
%{
unique_clusters = unique(idxs);
n_clusters = length(unique_clusters);
distances = nan(n_clusters);

for ii = 1:n_clusters
    for jj = 1:n_clusters
        distances(ii, jj)
        
    end
end
%}
    