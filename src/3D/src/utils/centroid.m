function C = centroid(mask)
% finds the centroid of a binary mask

T = regionprops(mask, 'centroid');
C = T.Centroid;