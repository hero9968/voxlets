function [stacked_img, stacked_img_cropped, transformed] = ...
    aggregate_masks(transforms, height, depth, params)

% settuping up variables
width = length(depth);
padding = 20;
N = length(transforms);
adding = false;
transform_type = params.transform_type; %'icp';
weights = ones(1, N) / 5;
known_mask = fill_grid_from_depth(depth, height, 0.5);

% apply transformations to the input images
transformed_masks = nan(height + 2*padding, width + 2*padding, N);

% sepcifying the range in the output image that we want to fill
x_data = [-padding+1, width + padding];
y_data = [-padding+1, height + padding];

% creating the stack of transformed masks
for ii = 1:N
    
    this_mask = +(transforms(ii).image > 0);
    this_transform = transforms(ii).(transform_type);
    this_depth = transforms(ii).depth;

    check_isgood_transform(this_transform);
    
    T = maketform('projective', this_transform');
    [transformed(ii).masks, transformed(ii).x_data, transformed(ii).y_data] = ...
        imtransform(this_mask, T, 'nearest', 'XYScale',1, 'XData', x_data, 'YData', y_data);
    transformed(ii).masks = +(transformed(ii).masks > 0);
       
    assert(range(transformed(ii).x_data)==range(x_data));
    assert(range(transformed(ii).y_data)==range(y_data));
    
    transformed_masks(:, :, ii) = transformed(ii).masks;
    transformed(ii).cropped_mask = transformed(ii).masks(padding+1:end-padding, padding+1:end-padding);
    transformed(ii).padding = padding;
    
    temp_traced = raytrace_2d(transformed(ii).cropped_mask);
    filled = fill_grid_from_depth(temp_traced, height, 0.5);
    filled(:, isnan(temp_traced)) = 1;
    final = (transformed(ii).cropped_mask | filled == 0) & known_mask~=0;
    transformed(ii).extended_mask = final;
  
    
    transformed(ii).depth = this_depth;
    XY = [1:length(this_depth); this_depth];
    temp_transform = [1, 0, padding; 0, 1, padding; 0, 0, 1] * this_transform;
    transformed(ii).transformed_depth = apply_transformation_2d(XY, temp_transform);
    
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
if params.apply_known_mask
    stacked_img_cropped(known_mask==0) = 0;
    stacked_img_cropped(known_mask==1) = 1;
end




function check_isgood_transform( trans )

if any(isnan(trans(:))) 
    disp('Seems like the transform is not very nice')
    keyboard
end

if cond(trans') > 1e8
    disp('Seems like conditioning is bad')
    keyboard
end

