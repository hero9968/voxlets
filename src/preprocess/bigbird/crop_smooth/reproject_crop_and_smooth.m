function cropped = reproject_crop_and_smooth(bb)

gray_image = im2double(bb.rgb);

% projecting the depth into the colour image space
reproj_depth = reproject_depth_into_im(bb.rgb, bb.depth, bb.K_depth, bb.K_rgb, bb.H_ir', bb.H_rgb');

% finding the aabb to crop to...
offset = 100;

H = size(bb.mask, 1);
W = size(bb.mask, 2);

x_lim = any(bb.mask, 1);
y_lim = any(bb.mask, 2);

left = max(find(x_lim, 1, 'first') - offset, 1);
right = min(find(x_lim, 1, 'last') + offset, W);
top = max(find(y_lim, 1, 'first') - offset, 1);
bottom = min(find(y_lim, 1, 'last') + offset, W);

cropped.aabb = [left, right, top, bottom];

% cropping the temporary image
reproj_depth_crop = reproj_depth(top:bottom, left:right);
reproj_depth_crop(isnan(reproj_depth_crop)) = 0;

% cropping the mask, rgb and grey images
cropped.grey = gray_image(top:bottom, left:right);
cropped.rgb = bb.rgb(top:bottom, left:right, :);
cropped.mask = bb.mask(top:bottom, left:right);

% now filling in the holes present in this new image
equ_grey = histeq(cropped.grey);
equ_grey(cropped.mask==0) = 1;
cropped.depth = fill_depth_colorization_ft_mex(equ_grey, reproj_depth_crop);
cropped.orig_d = reproj_depth_crop;

%temp_front = imresize(bb.front_render, 2 * size(bb.front_render));
%temp_back = imresize(bb.back_render, 2 * size(bb.front_render));
cropped.front_render = bb.front_render(top:bottom, left:right);
cropped.back_render = bb.back_render(top:bottom, left:right);

% finally adding on the transforms...
cropped.T.K_rgb = bb.K_rgb;
cropped.T.K_depth = bb.K_depth;
cropped.T.H_rgb = bb.H_rgb;
cropped.T.H_ir = bb.H_ir;
