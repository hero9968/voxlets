function [edges, rgb_edges] = depth_edges(rgb, depth)
% function to find the depth eges from an image

threshold = 0.05;
offset_dist = 5;
%%
%rgb  =img.rgb;
%depth = img.depth;

%% find edge and angles
grey = rgb2gray(rgb);
rgb_edges = edge(grey, 'canny');
[~, dir] = imgradient(grey);
dir = (dir / 180) * pi;

%% now choose which edges we want to use...

[I, J] = find(rgb_edges);

to_remove = I <= offset_dist | ...
    J <= offset_dist | ...
    I >= size(depth, 1) - offset_dist | ...
    J >= size(depth, 2) - offset_dist;

I(to_remove) = [];
J(to_remove) = [];

%% look up the edges in the system...
image_diff = zeros(size(depth));
N = length(I);
%imagesc2(grey);
%colormap(gray)
%hold on
idxs = sub2ind(size(depth), I, J);
t_dir = dir(idxs);

side_points(1).X = round(I + offset_dist .* sin(t_dir));
side_points(1).Y = round(J + offset_dist .* cos(t_dir));
side_points(2).X = round(I - offset_dist .* sin(t_dir));
side_points(2).Y = round(J - offset_dist .* cos(t_dir));

idxs1 = sub2ind(size(depth), side_points(1).X, side_points(1).Y);
idxs2 = sub2ind(size(depth), side_points(2).X, side_points(2).Y);

side_points(1).depth = depth(idxs1);
side_points(2).depth = depth(idxs2);

image_diff(idxs) = abs(side_points(1).depth - side_points(2).depth);
edges = image_diff>threshold;


if 0
    %%
    clf
    imagesc(depth)
    hold on
    axis image
    plot(side_points(1).Y, side_points(1).X, '.r')
    plot(side_points(2).Y, side_points(2).X, '.g')
    hold off
    
end