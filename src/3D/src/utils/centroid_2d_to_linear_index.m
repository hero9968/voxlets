function linear_index =  centroid_2d_to_linear_index(centroid_2d, mask)
% converts a 2D index position on the mask to a single linear index.
% This index refers to the pixel on the mask closest to the 2D centroid...

[XX, YY] = find(mask);
[~, linear_index] = pdist2([YY, XX], centroid_2d, 'euclidean', 'smallest', 1);

% error check
if linear_index == 0 || isempty(linear_index)
    keyboard;
end