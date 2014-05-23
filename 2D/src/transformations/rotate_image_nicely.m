function im_out = rotate_image_nicely(im_in, angle)
% rotates an image and pads and puts in the center just like the raytracer
% does

[~, temp] = rotate_and_raytrace_mask(im_in, angle, 1);
im_out = temp{1};
