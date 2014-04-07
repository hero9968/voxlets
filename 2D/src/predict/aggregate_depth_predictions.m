function [output_image, output_image_cropped, transformed] = ...
    aggregate_depth_predictions(prediction, image_size)


assert(isfield(prediction, 'mask'));
assert(isfield(prediction, 'transform'));
assert(isfield(prediction, 'weight'));

adding = false;

% input checks
N = length(prediction);

padding = 20;

% apply transformations to the input images
transformed_masks = nan([image_size + 2*padding, N]);



x_data = [-padding+1, image_size(2) + padding];
y_data = [-padding+1, image_size(1) + padding];


for ii = 1:N
    
    this_mask = prediction(ii).mask;
    %row_sums = sum(this_mask, 2);
    %image_end = findfirst(row_sums, 1, 1, 'last');
    %this_mask(image_end+1:end, :) = [];
    %this_mask = this_mask==1;
    
    % translating masks
    %prediction_transforms{ii}'
    
    check_isgood_transform(prediction(ii).transform);
    
    
    T = maketform('projective', prediction(ii).transform');
    [transformed(ii).masks, transformed(ii).x_data, transformed(ii).y_data] = ...
        imtransform(this_mask, T, 'bilinear', 'XYScale',1, 'XData', x_data, 'YData', y_data);
    
    transformed_masks(:, :, ii) = transformed(ii).masks;
    
    transformed(ii).padding = padding;
    %subplot(4, 4, ii);
    %imagesc(transformed_masks{ii});
end

%{
transformed.x_data = x_data;
transformed.y_data = y_data;
transformed.x_range = x_data(1):x_data(2);
transformed.y_range = y_data(1);y_data(2);
%}

if adding
    output_image = sum(transformed_masks, 3);
else
    output_image = noisy_or(transformed_masks, 3, [prediction.weight]);
end


output_image_cropped = output_image(padding:end-padding, padding:end-padding);




function check_isgood_transform( trans )

if any(isnan(trans(:))) 
    %|| ...
    %abs(abs(det(prediction_transforms{ii}))-1) > 0.0001
    disp('Seems like the transform is not very nice')
    keyboard
end

if cond(trans') > 1e7
    disp('Seems like conditioning is bad')
    keyboard
end

%{
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

%individual_images = cell(1, N);

for ii = 1:N
    % checking the size of the image lines up with the x and y data
    %{
    x_diff = abs(round(x_data{ii}(1)) - round(x_data{ii}(2))) +1;
    assert(x_diff == size(transformed_masks{ii} , 2))
    y_diff = abs(round(y_data{ii}(1)) - round(y_data{ii}(2))) +1;
    assert(y_diff == size(transformed_masks{ii} , 1))
    
    % forming the destination ranges
    x_range = (round(x_data{ii}(1)-x_min):round(x_data{ii}(2)-x_min)) +1;
    y_range = (round(y_data{ii}(1)-y_min):round(y_data{ii}(2)-y_min)) +1;
    %}
    x1 = round(x_data{ii}(1) - x_min) + 1;
    y1 = round(y_data{ii}(1) - y_min) + 1;
    x_range = x1:(x1 + size(transformed_masks{ii}, 2) - 1);
    y_range = y1:(y1 + size(transformed_masks{ii}, 1) - 1);
    
    to_add = prediction_weights(ii) * double(transformed_masks{ii});
    
    if adding
        output_image(y_range, x_range) = output_image(y_range, x_range) + to_add;
    else % soft or
        %output_image(y_range, x_range) = output_image(y_range, x_range) .* (1-to_add*0.3/prediction_weights(ii)); 
        output_image(y_range, x_range) = output_image(y_range, x_range) .* (1-to_add*prediction_weights(ii)); 
    end
    
    this_image_to_add = zeros(size(output_image));
    this_image_to_add(y_range, x_range) = transformed_masks{ii};
    individual_images.imgs{ii} = this_image_to_add;
        
end

if ~adding
     output_image = 1-output_image;
end

x_data = floor(x_min):(ceil(x_max)+1);
y_data = floor(y_min):(ceil(y_max)+1);

individual_images.x_range = x_data;%[x_min, x_max];
individual_images.y_range = y_data;%[y_min, y_max];
%}

%assert(length(x_data)==size(output_image, 2))
%assert(length(y_data)==size(output_image, 1))

