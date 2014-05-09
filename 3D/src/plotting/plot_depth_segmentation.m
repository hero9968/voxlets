function plot_depth_segmentation(depth_image, segmentation_image)
% function to nicely plot a single segment on a depth image

% check they're both the same size
assert(all(size(depth_image)==size(segmentation_image)));

imagesc(depth_image);
hold on

[X, Y] = find(edge(logical(segmentation_image)));
plot(Y, X, 'r.')
hold off

axis image off
colormap(flipud(gray))
    