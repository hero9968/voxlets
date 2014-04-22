function [mask_out, transform] = rotate_mask(mask, angle, params)
% takes a binary mask and an angle in degrees
% also crops the mask
% need to have as an output the transformation which takes the original
% image to this new transform...

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% doing it the old way...
mask_out = imrotate(mask, angle);

% remove columns 
column_sums = sum(mask_out, 1);
mask_out(:, column_sums==0) = [];

% remove rows between the object and the camera
row_sums = sum(mask_out, 2)';
[~, end_idx] = find(row_sums, 1, 'first');
mask_out(1:(end_idx-1), :) = [];

% resizeing to the correct width
mask_out = imresize(mask_out, params.scale);

% padding to the correct height
if size(mask_out, 1) < params.im_min_height
    padsize = params.im_min_height - size(mask_out, 1);
    mask_out = padarray(mask_out, [padsize, 0], 0, 'post');
else
    % do nothing - allow the mask to be bigger
    %mask_out = mask_out(1:params.im_min_height, :);
end

%assert(size(mask_out, 1) == params.im_height);
