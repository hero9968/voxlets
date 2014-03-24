function mask_out = rotate_mask(mask, angle, params)
% takes a binary mask and an angle in degrees
% renders the mask from the south rotated to the specified angle
% also crops the mask

mask_out = imrotate(mask, angle);

% remove columns 
column_sums = sum(mask_out, 1);
mask_out(:, column_sums==0) = [];

% remove rows below the bottom line
row_sums = sum(mask_out, 2)';
[~, end_idx] = find(row_sums, 1, 'last');
mask_out(end_idx+1:end, :) = [];

% resizeing to the correct width
%scale = params.im_width / size(mask_out, 2);
scale = params.scale;
mask_out = imresize(mask_out, scale);

% padding to the correct height
if size(mask_out, 1) < params.im_height
    padsize = params.im_height - size(mask_out, 1);
    mask_out = padarray(mask_out, [padsize, 0], 0, 'pre');
else
    mask_out = mask_out(end-params.im_height:end, :);
end

assert(size(mask_out, 1) == params.im_height);
%assert(size(mask_out, 2) == params.im_width);


