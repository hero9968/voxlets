function im_out = transform_image(im_in, transform)
% function to apply a transformation to an image
% basically a wrapper for imtransform

% getting image height and width
imheight = size(im_in, 1);
imwidth = size(im_in, 2);

diag_size = sqrt(imheight^2 + imwidth^2);

% applying transformation
%im_out = imtransform(im_in, transform, 'bilinear', 'XYScale', 1, ...
%        'xdata', [1, diag_size + 2], 'ydata', [1, diag_size + 2]);
im_out = myimtransform(im_in, transform, diag_size + 2, diag_size + 2);