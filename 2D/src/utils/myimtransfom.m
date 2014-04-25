%function im_out = myimtransform(im_in, T, width_out, height_out)
% my attempt at a simplified and faster imtransform.

im_in = imresize(rgb2gray(imread('peppers.png')), 0.4);


subplot(141);
imagesc(im_in); 
axis image; 
set(gca, 'clim',[0. 255])

T = rotation_matrix(30);
width_out = 1500;
height_out = 150;

tic
width_in = size(im_in, 1);
height_in = size(im_in, 2);

% setting up output image as a vector
im_out = zeros(height_out*width_out, 1);

% getting the poisitions of each of the pixels in the output image
[X, Y] = meshgrid(1:width_out, 1:height_out);
coords = [X(:), Y(:), ones(height_out*width_out, 1)];

% applying transform and converting from hom coords
transformed_coords_hom = inv(T) * coords';
transformed_coords = transformed_coords_hom(1:2, :) ./ repmat(transformed_coords_hom(3, :), [2, 1]);

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
im_out = uint8(reshape(im_out, height_out, width_out));
my_time = toc

subplot(142); 
imagesc(im_out);
colormap(gray)
set(gca, 'clim',[0. 255])
axis image

%now doing the matlab way

tic

Tmat = maketform('affine', T');
im_out_mat = imtransform(im_in, Tmat, 'nearest', 'xdata', [1, width_out], 'ydata', [1, height_out]);
matlab_time = toc

subplot(143); 
imagesc(im_out_mat);
colormap(gray)
set(gca, 'clim',[0. 255])
axis image

subplot(144); 
imagesc(abs(im_out - im_out_mat)>2)
axis image

ratio = matlab_time/my_time


%% reshaping output image



