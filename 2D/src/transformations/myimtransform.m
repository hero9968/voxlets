function [im_out, corners_transformed] = myimtransform(im_in, T_in, width_out, height_out)
% my attempt at a simplified and faster imtransform.

width_in = size(im_in, 1);
height_in = size(im_in, 2);

width_out = round(width_out);
height_out = round(height_out);


T = T_in;

% transforming the bounding box into the new image
corners = [0, 0; height_in, 0; height_in, width_in; 0, width_in]';
corners_transformed = apply_transformation_2d(corners, T, 'affine');

% forming an axis-aligned bounding box from this transformed bounding box
AABB = form_aabb(corners_transformed);

% truncating AABB to the dimensions of the output image
AABB = truncate_aabb(AABB, width_out, height_out);

% position of top left of transformed image in the output space
T(1, 3) = T(1, 3) - AABB.left;
T(2, 3) = T(2, 3) - AABB.top;

% setting up output image as a vector
im_out = zeros(AABB.height*AABB.width, 1);

% getting the poisitions of each of the pixels in the output image
[X, Y] = meshgrid(1:AABB.width, 1:AABB.height);

% applying the transformation to these values
transformed_coords = apply_transformation_2d([X(:)'; Y(:)'], inv(T), 'affine');

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
im_out = uint8(reshape(im_out, AABB.height, AABB.width));

% padding the removed pixels
im_out = padarray(im_out, [AABB.top, AABB.left], 0, 'pre');

top_padsize = max(0, height_out - AABB.height - AABB.top);
right_padsize =  max(0, width_out - AABB.width - AABB.left);

im_out = padarray(im_out, [top_padsize, right_padsize], 0, 'post');


function AABB = form_aabb(corners_transformed)
% forming an axis-aligned bounding box from the rotated corners

AABB.top = min(corners_transformed(2, :));
AABB.bottom = max(corners_transformed(2, :));
AABB.left = min(corners_transformed(1, :));
AABB.right = max(corners_transformed(1, :));


function AABB = truncate_aabb(AABB, width_out, height_out)
% truncating the axis-aligned bounding box according to the output dimensions

AABB.left = round(max(1, AABB.left));
AABB.top = round(max(1, AABB.top));

AABB.right = round(min(width_out, AABB.right));
AABB.bottom = round(min(height_out, AABB.bottom));
AABB.bottom = max(AABB.bottom, 1); % just in case its negative
AABB.right = max(AABB.right, 1); % just in case its negative

AABB.width = max(AABB.right - AABB.left, 0);
AABB.height = max(AABB.bottom - AABB.top, 0);

