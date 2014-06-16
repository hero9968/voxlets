function combined = combine_mask_and_depth(mask, depth)
% function to combine a mask image and a depth vector
% will plot the combined image unless an output argument is taken

assert(isvector(depth))
%assert(length(depth) == size(mask, 2));
difference = abs(length(depth)-size(mask, 2));
if difference > 0 && difference < 4
    warning('Depth image and rotated image not the same size - but close');
    if length(depth) < size(mask, 2)
        mask = mask(:, 1:length(depth));
    else
       mask = [mask, zeros(size(mask, 1), difference)];
    end
elseif difference > 4
    error('Big difference between depth and rotated image');
end

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