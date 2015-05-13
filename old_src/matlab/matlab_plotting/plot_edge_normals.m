function plot_normals_2d(XY, norms)
% 

assert(size(XY, 1) == 2);
assert(size(norms, 1) == 2);
assert(size(XY, 2) == size(norms, 2))

n = length(XY);
alpha = n / 20;

% plotting 2d points
plot(XY(1, :), XY(2, :), 'ob')
axis image
hold on

% plotting lines
X_n = repmat(XY(1, :), [2, 1]) + [zeros(1, n); alpha * norms(1, :)];
Y_n = repmat(XY(2, :), [2, 1]) + [zeros(1, n); alpha * norms(2, :)];
line(X_n, Y_n, 'Color', 'r', 'LineWidth', 1.5);

% setting axes
set(gca, 'xlim', [min(XY(1, :)) - alpha, max(XY(1, :)) + alpha])
set(gca, 'ylim', [min(XY(2, :)) - alpha, max(XY(2, :)) + alpha])


hold off
