function bb = reproject_crop_and_smooth(bb)

gray_image = im2double(bb.rgb);

% projecting the depth into the colour image space
reproj_depth = reproject_depth_into_im(bb.rgb, bb.depth, bb.K_depth, bb.K_rgb, bb.H_ir', bb.H_rgb');

% now cropping this down using the mask
offset = 50;

H = size(bb.mask, 1);
W = size(bb.mask, 2);

x_lim = any(bb.mask, 1);
y_lim = any(bb.mask, 2);

left = max(find(x_lim, 1, 'first') - offset, 1);
right = min(find(x_lim, 1, 'last') + offset, W);
top = max(find(y_lim, 1, 'first') - offset, 1);
bottom = min(find(y_lim, 1, 'last') + offset, W);

bb.aabb = [left, right, top, bottom];

reproj_depth_crop = reproj_depth(top:bottom, left:right);
reproj_depth_crop(isnan(reproj_depth_crop)) = 0;

gray_image_crop = gray_image(top:bottom, left:right);

% now filling in the holes present in this new image
bb.depth_smooth_crop = fill_depth_colorization_ft_mex(gray_image_crop, reproj_depth_crop);
