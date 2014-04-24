function model = gaussian_model_train(images, train_data, params)
% train the parameters of a gaussian model
% split the shape into equally width bins. Model each bin individually.
% model final point in each bin with a gaussian. at test time use the
% comulative distribution to model occupancy

num_bins = params.gauss_model.number_bins;

max_depth_to_use = 5; % max depth to take into account relative to the width of the image

% input checks
assert(iscell(images))
N = length(train_data);

% now want to accumulate the distances to the back of each image for each
% input image
per_bin_depths = cell(num_bins, N);

for ii = 1:N

    % width of the raytraced image
    this_depth = single(train_data(ii).depth);
    depth_width = length(this_depth);
    
    % find points outside bounds of the raytraced image
    to_remove = outside_nans(this_depth);
    
    % getting the ground truth image
    this_image = images{train_data(ii).image_idx};
    this_transform = train_data(ii).transform;
    im_transformed = ...
        imtransform(this_image, this_transform, ...
                    'xdata', [1, depth_width], 'ydata', [1, depth_width]);
    
    % finding the minimum and maxium depths    
    min_depth = this_depth;
    max_depth = findfirst(im_transformed, 1, 1, 'last');
    overall_depth = max_depth - min_depth;
    
    % removing the nan points on the outside of the image
    overall_depth(to_remove) = [];
    width = length(overall_depth);
       
    % scale to fixed width
    overall_depth_scaled = (overall_depth / width);
    
    overall_depth_scaled(overall_depth_scaled > max_depth_to_use) = nan;
    
    bin_idxs = ceil(((1:width)/width)*num_bins);
        
    % now accumulate the depths into bins
    per_bin_depths(:, ii) = accumarray(bin_idxs', overall_depth_scaled', [num_bins, 1], @(x)({x}));
       
    done(ii, N, 100);
end

model.per_bin_depths = per_bin_depths;

% now combine all the depths for each bin
for ii = 1:num_bins
    
    these_cells = per_bin_depths(ii, :);
    empty_cells = cellfun(@isempty, these_cells);
    these_cells(empty_cells) = [];
    
    % get all the depths for this bin
    all_depths = cell2mat(these_cells');
    all_depths(isnan(all_depths)) = [];
    
    % model depths with a gaussian
    model.means(ii) = mean(all_depths);
    model.stds(ii) = std(all_depths);
    
    %subplot(4, 5, ii)
    %bar(histc(all_depths, 0:0.1:1.5))
    
end

model.num_bins = num_bins;

% predict a vector of depths for each bin width


