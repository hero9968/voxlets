function mask_out = per_ray_gaussian_prediction( depth_in, model, params )
% a function to predict the occupancy mask given the depth image in
% this function will use a very simple gaussian fall-off.

width = length(depth_in);
height = params.im_height;

% fill out mask with the depth we can see
mask_out = zeros(height, width);

for ii = 1:width
    if depth_in(ii) > 0
        mask_out(depth_in(ii), ii) = 1;
    end
end

% now fill in a gaussian tail from each ground truth prediction...
% ( ultimately would like to learn these values - given the position across
% the image in percent, what is the typidal fall-off...)
full_height = 1:params.im_height;

gaussian_profile = normpdf(full_height, model.mu, model.sigma);
gaussian_profile = gaussian_profile / gaussian_profile(1);
%plot(full_height, gaussian_profile);

for ii = 1:width
    if depth_in(ii) > 0
        mask_out(depth_in(ii):end, ii) = gaussian_profile(1:end-depth_in(ii)+1);
    end
end



