function plot_depth(depth)

max_depth = max(depth(:));
depth(abs(depth-max_depth)<0.01) = nan;
imagesc(depth)
axis image off
colormap(flipud(gray))