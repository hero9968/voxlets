function plot_depth_segmentation(depth_image, segmentation_image)
% function to nicely plot a single segment on a depth image

% check they're both the same size
assert(size(depth_image, 1)==size(segmentation_image, 1));
assert(size(depth_image, 2)==size(segmentation_image, 2));

imagesc(depth_image);
hold on

[X, Y] = find(edge(logical(segmentation_image)));
plot(Y, X, 'r.')
hold off

axis image off
colormap(flipud(gray))
    