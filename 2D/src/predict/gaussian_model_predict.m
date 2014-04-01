function output_image = gaussian_model_predict(model, depth, params)

% input checks
assert(isvector(depth));
depth(isnan(depth)) = 1;

width = length(depth);

% loop over each bin and predict a vector of depths for it
bin_vectors = nan(params.im_height, model.num_bins);
x_values = 1:params.im_height;

for ii = 1:model.num_bins
    
    this_sigma = model.stds(ii) * width;
    this_mean = model.means(ii) * width;
    
    this_prediction = 1 - normcdf(x_values, this_mean, this_sigma);
    
    bin_vectors(:, ii) = this_prediction;
    
end

%imagesc(bin_vectors);
bin_idxs = ceil(((1:width)/width)*model.num_bins);
output_image = zeros(params.im_height, width);

% now fill in output image
for ii = 1:width
    
    this_bin = bin_idxs(ii); 
    
    temp_bin_vector = bin_vectors(1:end-depth(ii)+1, this_bin);
        
    output_image(depth(ii):end, ii) = temp_bin_vector;
        
end

% pad in the first row with ones
rendered_depth = fill_grid_from_depth(depth, params.im_height, 0);
output_image(rendered_depth == 1) = 1;

