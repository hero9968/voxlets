function [trans, rot, combined_T] = transformation_to_origin_2d(XY)
% compute the mean and eigenvalues of points XY.

assert(size(XY, 1) == 2);
XY = XY';
%XY = [X(:), Y(:)];

to_remove = any(isnan(XY), 2);
XY(to_remove, :) = [];

trans = mean(XY, 1);

[rot, dummy] = eig(cov(XY));

% fixing for incorrect sign
if det(rot) < 0
    rot(:, 1) = -rot(:, 1);
end

% fixing for normal in wrong direction
idx = find(diag(dummy)==min(diag(dummy)));
normal = rot(:, idx);
if normal(2) < 0
    rot(:, idx) = -rot(:, idx);
end


% now combining into a 3x3 matrix
combined_T = [rot, trans'; 0 0 1];
