function  plotNormals( XYZ, norms, fraction, alpha )
%PLOTNORMALS plots the points and normals in a niceish way
%
%USAGE
% plotNormals( XYZ, norms, fraction )
%
% ARGUMENTS
%   XYZ -   3 x n x m matrix of normal vectors
%   norms - 3 x n x m matrix of normal vectors
%
% EXAMPLE
%  plotNormals([X,Z,Y]', reshape_cloud_matrix(imgNormals)', 0.01)
%
% todo - option to plot as image instead? using imagesc
% todo - arrows on normals?

%if size(XYZ, 3) == 3
%  XYZ = permute(XYZ, [3, 1, 2]);
  
%end

%norms = permute(norms, [3, 1, 2]);
assert(size(XYZ, 1) == size(norms, 1))


%XYZ = XYZ(:, :);
%norms = norms(:, :);

% scaling factor for normals
%alpha = 0.005;
%alpha = 0.1;
% estimate a scaling factor
if nargin < 4
    alpha = 0.025*max(range(XYZ));
end


% plotting points
%plot3(XYZ(1, :), XYZ(2, :), XYZ(3, :), '.', 'MarkerSize', 7);
plot3d(XYZ, [0 0 1]);
n = length(XYZ);

% plotting lines
X = repmat(XYZ(:, 1), [1, 2]) + [zeros(n, 1), alpha * norms(:, 1)];
Y = repmat(XYZ(:, 2), [1, 2]) + [zeros(n, 1), alpha * norms(:, 2)];
Z = repmat(XYZ(:, 3), [1, 2]) + [zeros(n, 1), alpha * norms(:, 3)];

if nargin == 3
    number_points = round(fraction * n);
    inds = randperm(n, number_points);
    X = X(inds, :);
    Y = Y(inds, :);
    Z = Z(inds, :);
end

line(X', Y', Z', 'Color', 'r', 'LineWidth', 1.5);

axis equal

end

