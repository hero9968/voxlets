function [trans, rot, combined_T] = transformation_to_origin_3d(XYZ)
% compute the mean and eigenvalues of points XY.

assert(size(XYZ, 2) == 3);

to_remove = any(isnan(XYZ), 2);
XYZ(to_remove, :) = [];

trans = mean(XYZ, 1);

[rot, dummy] = eig(cov(XYZ));

% fixing for incorrect sign
if det(rot) < 0
    rot(:, 1) = -rot(:, 1);
end

% fixing for normal in wrong direction
idx = find(diag(dummy)==min(diag(dummy)));
normal = rot(:, idx);
if normal(3) < 0
    rot(:, idx) = -rot(:, idx);
end


% now combining into a 3x3 matrix
combined_T = [rot, trans'; 0 0 0 1];
