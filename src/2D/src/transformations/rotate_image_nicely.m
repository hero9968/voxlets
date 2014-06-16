function im_out = rotate_image_nicely(im_in, angle, max_width)
% rotates an image and pads and puts in the center just like the raytracer
% does

[~, temp] = rotate_and_raytrace_mask(im_in, angle, 1);
im_out = temp{1};

if nargin == 3
    if size(im_out, 2) == max_width
        disp('Correct width')
    elseif abs(size(im_out, 2)- max_width) < 3
        warning('Wrong width but only by couple of pixels')
        if size(im_out, 2) < max_width
            error('% pad image here')
        else
            im_out = im_out(:, 1:max_width);
        end
    else
        error('Very different widths')
    end
end
