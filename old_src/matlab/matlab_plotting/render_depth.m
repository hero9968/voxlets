function render_depth(depth, max_depth)

if nargin == 2
    max_depth_idx = abs(depth-max_depth) < 0.0001;
    depth(max_depth_idx) = nan;
end

h = imagesc(depth);
axis image off

if nargin == 2
    set(h, 'AlphaData', ~max_depth_idx)
end