function linear_index =  centroid_2d_to_linear_index(centroid_2d, mask)
% converts a 2D index position on the mask to a single linear index.
% This index refers to the pixel on the mask closest to the 2D centroid...

[XX, YY] = find(mask);
[~, idx] = pdist2([YY, XX], centroid_2d, 'euclidean', 'smallest', 1);

temp_mask = +mask;
temp_mask(mask) = 1:sum(sum(mask));

linear_index = temp_mask(round(centroid_2d(2)), round(centroid_2d(1)));