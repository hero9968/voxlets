function stacked_image = full_aggregation(transforms, height, depth)

width = length(depth);


% now creating the combined image
prediction_masks = {transforms.image};
prediction_transformations = {transforms.icp};
prediction_weights = ones(1, length(transforms)) / length(transforms);
[stacked_image_full, x_data, y_data] = ...
    aggregate_depth_predictions(prediction_masks, prediction_transformations, prediction_weights);

% cropping and resizing the image
x_range = x_data > 0 & x_data <= width;
y_range = y_data > 0 & y_data <= height;
stacked_image_cropped = stacked_image_full(y_range, x_range);

if isempty(stacked_image_cropped)
    stacked_image_cropped = zeros(height, width);
else
    pad_left_size = max(0, x_data(1) - 1 );
    stacked_image_cropped = [zeros(size(stacked_image_cropped, 1), pad_left_size), stacked_image_cropped];

    pad_right_size = max(0, width - size(stacked_image_cropped, 2));
    stacked_image_cropped = [stacked_image_cropped, zeros(size(stacked_image_cropped, 1), pad_right_size)];

    extra_height = height - size(stacked_image_cropped, 1);
    stacked_image_cropped = [stacked_image_cropped; zeros(extra_height, width)];
end

% applying the known mask for the known free pixels
known_mask = fill_grid_from_depth(depth, height, 0.5);
stacked_image_cropped(known_mask==0) = 0;
stacked_image_cropped(known_mask==1) = 1;


stacked_image = stacked_image_cropped;