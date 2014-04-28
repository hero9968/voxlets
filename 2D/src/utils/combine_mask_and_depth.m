function combined = combine_mask_and_depth(mask, depth)
% function to combine a mask image and a depth vector
% will plot the combined image unless an output argument is taken

assert(isvector(depth))
assert(length(depth) == size(mask, 2));

image_height = size(mask, 1);

depth_image = fill_grid_from_depth(depth, image_height, 0);

if size(depth_image, 1) > size(mask, 1)
    depth_image = depth_image(1:size(mask, 1), :);
end

combined = double(mask)/255 + 3 * depth_image;

% plotting if no output arguments
if nargout == 0
    imagesc(combined)
	axis image
	colormap(flipud(gray))
    clear combined
end