K1 = rand(3);
K2 = rand(3);
H1 = rand(4);
H2 = rand(4);

x = [100, 200];

camera_to_camera(x', 10, K1, K2, H1, H2)
clear


addpath('toolbox_nyu_depth_v2/')

%% loading the data
base = '/Users/Michael/projects/shape_sharing/data/bigbird/3m_high_tack_spray_adhesive/';
obj_name = 'NP3_144';
obj = [base, obj_name];

% loading the depth and rgb
depth = h5read([obj, '.h5'], '/depth');
imgDepthAbs = double(depth') / 10000;
imgRgb = imread([obj, '.jpg']);

% loading the intrinsics
K2 = h5read([base, 'calibration.h5'], '/NP3_rgb_K')';
K1 = h5read([base, 'calibration.h5'], '/NP3_depth_K')';

% loading the extrinsics
H2 = h5read([base, 'calibration.h5'], '/H_NP3_from_NP5');
H1 = h5read([base, 'calibration.h5'], '/H_NP3_ir_from_NP5');

x = repmat([100; 200], 1, 100);
camera_to_camera(x, rand(1, size(x, 2)), K1, K2, H1, H2)

% plotting
h1 = subplot(121)
imagesc(imgDepthAbs)
axis image

h2 = subplot(122)
imagesc(imgRgb)
axis image


%% endless loop reprjecting points
while 1
    % get a point from h1
    axes(h1)
    [x, y] = ginput(1);
    depth = imgDepthAbs(round(y), round(x));
    hold on
    plot(x, y, '+', 'Markersize', 10)
    hold off
    
    % project to h2
    cam2_point = camera_to_camera([x, y], depth, K1, K2, H1', H2');
    
    % plot on h2
    axes(h2)
    hold on
    plot(cam2_point(1), cam2_point(2), '+r', 'Markersize', 10)
    hold off
    
end

%% faster reprojection
[uu, vv] = meshgrid(1:size(imgDepthAbs, 2), 1:size(imgDepthAbs, 1));
stack = [vv(:), uu(:)]';
imgDepthAbs2 = imgDepthAbs';
projected_points = camera_to_camera(stack, imgDepthAbs(:)', K1, K2, H1', H2');

%%
to_remove = projected_points(1, :) < 1 | projected_points(2, :) < 1 | ...
    projected_points(1, :) > size(imgRgb, 1) | projected_points(2, :)  > size(imgRgb, 2) ...
    | any(isnan(projected_points), 1);

new_projected_points = round(projected_points(:, ~to_remove));


%%
imgGrey = rgb2gray(imgRgb);
idxs = sub2ind(size(imgGrey), new_projected_points(1, :), new_projected_points(2, :));
extracted = imgGrey(idxs);

output_image = nan(size(imgDepthAbs));
output_image(find(~to_remove)) = extracted;

imagesc(output_image);
axis image


%%

h1 = subplot(131)
imagesc(imgDepthAbs)
axis image

h2 = subplot(132)
imagesc(uint8(reprojected_rgb))
axis image

h1 = subplot(133)
imagesc(imgDepthAbs + rgb2gray(reprojected_rgb))
axis image
%% reprojecting color into depth
reprojected_rgb = zeros([size(imgDepthAbs), 3]);

for ii = 1:size(imgDepthAbs, 1)
    for jj = 1:size(imgDepthAbs, 2)
        
        depth = imgDepthAbs(ii, jj);
        projected_point = camera_to_camera([jj, ii]', depth, K1, K2, H1', H2');
        cam2_point = round(projected_point);
        
        if cam2_point(1) > 0 && cam2_point(1) < size(imgRgb, 1) && ...
                cam2_point(2) > 0 && cam2_point(2) < size(imgRgb, 2)
            
            reprojected_rgb(ii, jj, :) = imgRgb(cam2_point(2), cam2_point(1), :);

        end     
    end
    ii
end



%% reprojecting color into depth - by row
reprojected_rgb_r = zeros([size(imgDepthAbs), 3]);

for ii = 1:size(imgDepthAbs, 1)
           
    depth = imgDepthAbs(ii, :);
    jjs = 1:size(imgDepthAbs, 2);
    iis = repmat(ii, 1, size(imgDepthAbs, 2));
    stack = [jjs; iis];
    
    projected_point = camera_to_camera(stack, depth, K1, K2, H1', H2');
    
    cam2_point = round(projected_point);

    for jj = 1:length(cam2_point)
        if cam2_point(1, jj) > 0 && cam2_point(1, jj) < size(imgRgb, 1) && ...
                cam2_point(2, jj) > 0 && cam2_point(2, jj) < size(imgRgb, 2)

            reprojected_rgb_r(ii, jj, :) = imgRgb(cam2_point(2, jj), cam2_point(1, jj), :);
        end
    end     
    ii
end

%%
