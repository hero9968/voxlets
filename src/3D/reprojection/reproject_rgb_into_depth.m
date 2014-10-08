function reprojected_rgb_r = reproject_rgb_into_depth(imgRgb, depth_im, K1, K2, H1, H2)

reprojected_rgb_r = zeros([size(depth_im), 3]);

for ii = 1:size(depth_im, 1)
           
    depth = depth_im(ii, :);
    jjs = 1:size(depth_im, 2);
    iis = repmat(ii, 1, size(depth_im, 2));
    stack = [jjs; iis];
    
    projected_point = camera_to_camera(stack, depth, K1, K2, H1, H2);
    
    cam2_point = round(projected_point);

    for jj = 1:length(cam2_point)
        if cam2_point(1, jj) > 0 && cam2_point(1, jj) < size(imgRgb, 1) && ...
                cam2_point(2, jj) > 0 && cam2_point(2, jj) < size(imgRgb, 2)

            reprojected_rgb_r(ii, jj, :) = imgRgb(cam2_point(2, jj), cam2_point(1, jj), :);
        end
    end     
    ii
end