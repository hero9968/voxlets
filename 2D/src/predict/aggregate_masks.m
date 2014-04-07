function [stacked_img, stacked_img_cropped, transformed] = aggregate_masks(transforms, height, depth)

% settuping up variables
width = length(depth);
padding = 20;
N = length(transforms);
adding = false;
transform_type = 'icp';
weights = ones(1, N) / 40;

% apply transformations to the input images
transformed_masks = nan(height + 2*padding, width + 2*padding, N);

% sepcifying the range in the output image that we want to fill
x_data = [-padding+1, width + padding];
y_data = [-padding+1, height + padding];

% creating the stack of transformed masks
for ii = 1:N
    
    this_mask = transforms(ii).image;
    this_transform = transforms(ii).(transform_type);

    check_isgood_transform(this_transform);
    
    T = maketform('projective', this_transform');
    [transformed(ii).masks, transformed(ii).x_data, transformed(ii).y_data] = ...
        imtransform(this_mask, T, 'bilinear', 'XYScale',1, 'XData', x_data, 'YData', y_data);
    
    assert(range(transformed(ii).x_data)==range(x_data));
    assert(range(transformed(ii).y_data)==range(y_data));
    
    transformed_masks(:, :, ii) = transformed(ii).masks;
    transformed(ii).cropped_mask = transformed(ii).masks(padding+1:end-padding, padding+1:end-padding);
    transformed(ii).padding = padding;
    
end

% combine output images
if adding
    stacked_img = sum(transformed_masks, 3);
else
    stacked_img = noisy_or(transformed_masks, 3, weights);
end

% cropping output image
stacked_img_cropped = stacked_img(padding+1:end-padding, padding+1:end-padding);
assert(size(stacked_img_cropped, 1) == height);
assert(size(stacked_img_cropped, 2) == width);

% applying the known mask for the known free pixels
known_mask = fill_grid_from_depth(depth, height, 0.5);
stacked_img_cropped(known_mask==0) = 0;
stacked_img_cropped(known_mask==1) = 1;




function check_isgood_transform( trans )

if any(isnan(trans(:))) 
    disp('Seems like the transform is not very nice')
    keyboard
end

if cond(trans') > 1e7
    disp('Seems like conditioning is bad')
    keyboard
end

