function voxeldiff(v1, v2)
% plots a visual diff of two voxel volumes
% poitns coloured red and green like unix diff

v1 = double(round(v1));
v2 = double(round(v2));

v1_only = setdiff(v1, v2, 'rows');
v2_only = setdiff(v2, v1, 'rows');

clf
plot3d(v1_only, 'r');
hold on
plot3d(v2_only, 'g');
hold off

if isempty(v1_only) && isempty(v2_only)
    disp('No difference - inputs equal')
end