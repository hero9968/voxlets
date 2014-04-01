function model = gaussian_model_train(images, depths, params)
% train the parameters of a gaussian model
% split the shape into equally width bins. Model each bin individually.
% model final point in each bin with a gaussian. at test time use the
% comulative distribution to model occupancy

num_bins = params.gauss_model.number_bins;

% input checks
assert(iscell(images))
assert(iscell(depths))
assert(length(images)==length(depths));
N = length(images);

% now want to accumulate the distances to the back of each image for each
% input image
per_bin_depths = cell(num_bins, N);
for ii = 1:N

    width = length(depths{ii});
    
    min_depth = (double(depths{ii}));
    max_depth = findfirst(images{ii}, 1, 1, 'last');
    overall_depth = max_depth - min_depth;
       
    % scale to fixed width
    overall_depth_scaled = (overall_depth / width);
    
    bin_idxs = ceil(((1:width)/width)*num_bins);
        
    % now accumulate the depths into bins
    per_bin_depths(:, ii) = accumarray(bin_idxs', overall_depth_scaled', [num_bins, 1], @(x)({x}));
       

end

model.per_bin_depths = per_bin_depths;

% now combine all the depths for each bin
for ii = 1:num_bins
    
    % get all the depths for this bin
    all_depths = cell2mat([per_bin_depths(ii, :)']);
    all_depths(isnan(all_depths)) = [];
    
    % model depths with a gaussian
    model.means(ii) = mean(all_depths);
    model.stds(ii) = std(all_depths);
    
    %subplot(4, 5, ii)
    %bar(histc(all_depths, 0:0.1:1.5))
    
end

model.num_bins = num_bins;

% predict a vector of depths for each bin width


