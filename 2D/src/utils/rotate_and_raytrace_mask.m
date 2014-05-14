function [rotated, rotated_images] = rotate_and_raytrace_mask(mask, angles, scale)
% takes a binary mask and a set of angles in degrees
% returns a structure containing:
% - the original iamge
% - a resized and cropped image
% - a set of transformations to rotate and translate the cropped image
% - a set of raytraces of the image

plotting  = 0;%params.plotting.plot_rotated_masks;

% crop edges from image
rotated.original_image = mask;
rotated.image = boxcrop_2d(mask);

% resize image according to predefined scale
rotated.image = imresize(rotated.image, scale);

% getting image height and width
imheight = size(rotated.image, 1);
imwidth = size(rotated.image, 2);

% get the diagonal size of the image
diag_size = sqrt(imheight^2 + imwidth^2);

% initialise the rendered structure array
rendered(length(angles)) = ...
    struct('angle', [], 'transform', [], 'depth', []);

if nargout == 2
    rotated_images = cell(1, length(angles));
end

% loop over each possible rotation and 
for jj = 1:length(angles)

    % develop the transformation matrix
    T_translate_1 = translation_matrix(-imwidth/2, -imheight/2);
    T_rotate = rotation_matrix(angles(jj));
    T_translate_2 = translation_matrix(2 + diag_size/2, 2 + diag_size/2);
    % ^^ adding 2 to diag size to allow for rounding errors etc.

    % combining and converting to MATLAB format
    T_final = T_translate_2 * T_rotate * T_translate_1;
    rendered(jj).transform = maketform('affine', T_final');

    % applying transformation
    this_rotated_image = ...
        imtransform(rotated.image, rendered(jj).transform, ...
        'bilinear', 'XYScale', 1, ...
        'xdata', [1, diag_size + 2], ...
        'ydata', [1, diag_size + 2]);

    if nargout == 2
        rotated_images{jj} = this_rotated_image;
    end
    
    % raytracing this rotated image
    rendered(jj).depth = raytrace_2d(this_rotated_image > 128);                
    rendered(jj).angle = angles(jj);
    
    % plot this image
    if plotting
        [n, m] = best_subplot_dims(length(angles));
        subplot(n, m, jj)
        imagesc(this_rotated_image)
        axis image
        colormap(flipgray)
        hold on
        plot(1:length(rendered(jj).depth), rendered(jj).depth, 'linewidth', 3);
        hold off
    end

end

rotated.rendered = rendered;

