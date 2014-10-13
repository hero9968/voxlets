function reprojected_depth = reproject_depth_into_im(imgRgb, depth_im, K1, K2, H1, H2)

reprojected_depth = inf([size(imgRgb, 1), size(imgRgb, 2)]);

% do this row by row
for ii = 1:size(depth_im, 1)
           
    depth = depth_im(ii, :);
    jjs = 1:size(depth_im, 2);
    iis = repmat(ii, 1, size(depth_im, 2));
    stack = [jjs; iis];
    
    % project these points into the rgb image
    projected_point = camera_to_camera(stack, depth, K1, K2, H1, H2);
    
    cam2_point = round(projected_point);

    % check is in range...
    for jj = 1:length(cam2_point)
        if cam2_point(1, jj) > 0 && cam2_point(1, jj) < size(imgRgb, 2) && ...
                cam2_point(2, jj) > 0 && cam2_point(2, jj) < size(imgRgb, 1)

            % fill in image...
            reprojected_depth(cam2_point(2, jj), cam2_point(1, jj)) = ...
                min(reprojected_depth(cam2_point(2, jj), cam2_point(1, jj)), depth_im(ii, jj));
            
        end
    end     
    
end

reprojected_depth(isinf(reprojected_depth)) = nan;%./ count_im;