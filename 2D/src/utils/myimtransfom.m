function im_out = myimtransfom(im_in, T, width_out, height_out)
% my attempt at a simplified and faster imtransform.

width_in = size(im_in, 1);
height_in = size(im_in, 2);

% setting up output image as a vector
im_out = zeros(height_out*width_out, 1);

% getting the poisitions of each of the pixels in the output image
[X, Y] = meshgrid(1:width_out, 1:height_out);

% applying the transformation to these values
transformed_coords = apply_transformation_2d([X(:)'; Y(:)'], T');

% discovering their values in the original image
transformed_coords = round(transformed_coords);
in_range = transformed_coords(1, :) > 0 & ...
           transformed_coords(1, :) <= height_in & ...
           transformed_coords(2, :) > 0 & ...
           transformed_coords(2, :) <= width_in;

in_range_transformed_coords = transformed_coords(:, in_range);

% look up the colours in the original image
idx = sub2ind([width_in, height_in], in_range_transformed_coords(2, :), in_range_transformed_coords(1, :));
original_pixels = im_in(idx);

% now refilling the original pixels
im_out(in_range) = original_pixels;

% reshaping output image
im_out = uint8(reshape(im_out, height_out, width_out));






