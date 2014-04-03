function [output_image, x_data, y_data] = ...
    aggregate_depth_predictions(prediction_masks, prediction_transforms, prediction_weights)

adding = false;

% input checks
assert(iscell(prediction_masks))
assert(iscell(prediction_transforms))
assert(isvector(prediction_weights))
N = length(prediction_masks);
assert(N == length(prediction_transforms))
assert(N == length(prediction_weights))

% apply transformations to the input images
transformed_masks = cell(1, N);
x_data = cell(1, N);
y_data = cell(1, N);

for ii = 1:N
    
    this_mask = prediction_masks{ii};
    row_sums = sum(this_mask, 2);
    image_end = findfirst(row_sums, 1, 1, 'last');
    this_mask(image_end+1:end, :) = [];
    this_mask = this_mask==1;
    
    % translating masks
    %prediction_transforms{ii}'
    
    if any(isnan(prediction_transforms{ii}(:))) 
        %|| ...
        %abs(abs(det(prediction_transforms{ii}))-1) > 0.0001
        disp(['Seems like the transform is not very nice'])
        keyboard
    end
    
    if cond(prediction_transforms{ii}') > 1e7
        disp(['Seems like conditioning is bad'])
        keyboard
    end
    
    T = maketform('affine', prediction_transforms{ii}');
    [transformed_masks{ii}, x_data{ii}, y_data{ii}] = imtransform(this_mask, T, 'bilinear');
    
    if ii == 19
        %keyboard
    end
    
    %subplot(4, 4, ii);
    %imagesc(transformed_masks{ii});
end

% now stacking up all the transformed masks
x_min = min(cellfun(@min, x_data));
x_max = max(cellfun(@max, x_data));
y_min = min(cellfun(@min, y_data));
y_max = max(cellfun(@max, y_data));

if adding
    output_image = zeros(ceil(y_max)-floor(y_min)+2, ceil(x_max)-floor(x_min)+2);
else % soft or
    output_image = ones(ceil(y_max)-floor(y_min)+2, ceil(x_max)-floor(x_min)+2);
end


for ii = 1:N
    % checking the size of the image lines up with the x and y data
    x_diff = abs(round(x_data{ii}(1)) - round(x_data{ii}(2))) +1;
    assert(x_diff == size(transformed_masks{ii} , 2))
    y_diff = abs(round(y_data{ii}(1)) - round(y_data{ii}(2))) +1;
    assert(y_diff == size(transformed_masks{ii} , 1))
    
    % forming the destination ranges
    x_range = (round(x_data{ii}(1)-x_min):round(x_data{ii}(2)-x_min)) +1;
    y_range = (round(y_data{ii}(1)-y_min):round(y_data{ii}(2)-y_min)) +1;
    
    to_add = prediction_weights(ii) * double(transformed_masks{ii});
    
    if adding
        output_image(y_range, x_range) = output_image(y_range, x_range) + to_add;
    else % soft or
        output_image(y_range, x_range) = output_image(y_range, x_range) .* (1-to_add*0.3/prediction_weights(ii)); 
    end
        
end

if ~adding
     output_image = 1-output_image;
end

x_data = floor(x_min):(ceil(x_max)+1);
y_data = floor(y_min):(ceil(y_max)+1);


assert(length(x_data)==size(output_image, 2))
assert(length(y_data)==size(output_image, 1))

