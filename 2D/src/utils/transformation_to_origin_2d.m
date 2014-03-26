function [trans, rot, M] = get_trans_rot(X, Y)
% compute the mean and eigenvalues of points XY.

XY = [X(:), Y(:)];

trans = mean(XY, 1);

[rot, dummy] = eig(cov(XY));

% fixing for incorrect sign
if det(rot) < 0
    rot(:, 1) = -rot(:, 1);
end

% now combining into a 3x3 matrix
M = [rot, trans'; 0 0 1];
