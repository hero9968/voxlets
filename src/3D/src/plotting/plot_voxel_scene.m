function plot_voxel_scene(cloud, voxels, height)
% start of a 2-plot visualiser for voxel grids
% cloud - structr with xyz, depth

threshold = 0.0025;

temp_xyz = apply_transformation_3d(cloud.xyz, cloud.segment.rotate_to_plane);
pixel_height  = temp_xyz(:, 3);

subplot(121)
plot_line_scene(cloud.rgb, pixel_height, height, threshold);
title(num2str(height))

subplot(122)
cla
plot_voxel_slice(voxels, temp_xyz, height, threshold);



function plot_line_scene(rgb, pixel_height, desired_height, threshold)
% pixel_height is a NxM image of the height of each pixel above the ground plane
% rgb is NxMx3 RGB image...
% threshold is the width above and below we can look at
% desired_height is the height at which we are interested in

mask.at_line = abs(pixel_height - desired_height) < threshold;
mask.below_line = pixel_height - threshold <  desired_height;

line_col = [1, 0, 0];

% modify each channel of colour image one at a time - 
for ii = 1:3
    temp = rgb(:, :, ii);
    temp(mask.at_line) = line_col(ii);
    temp(mask.below_line) = temp(mask.below_line).^0.5;
    rgb(:, :, ii) = temp;
end

imagesc(rgb);
axis image off



function plot_voxel_slice(voxels, xyz, height, threshold)
% plot a slice from a voxel plot

voxels_to_plot = abs(voxels(:, 3)/100 - height) < threshold;
xyz_to_plot = abs(xyz(:, 3) - height) < threshold;

sum(voxels_to_plot);
sum(xyz_to_plot);

scatter(voxels(voxels_to_plot, 2)/100, -voxels(voxels_to_plot, 1)/100, voxels(voxels_to_plot, 4)*10, 'bo');
hold on
scatter(xyz(xyz_to_plot, 2), -xyz(xyz_to_plot, 1), 2, 'r+');%, 'filled');
hold off
axis image

%xmax = max(abs(get(gca, 'xlim')));
xmax = max(abs(xyz(:, 2)));
set(gca, 'xlim', [-xmax, xmax])
set(gca, 'ylim', [-max(xyz(:, 1)), -min(xyz(:, 1))])