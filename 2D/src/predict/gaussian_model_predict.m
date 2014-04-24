function output_image = gaussian_model_predict(model, depth, im_height)

% input checks
assert(isvector(depth));
%depth(isnan(depth)) = 1;

% strip nans from outside of depth
[to_remove] = outside_nans(depth);
cropped_width = sum(~to_remove);
original_width = length(depth);

% loop over each bin and predict a vector of depths for it
bin_vectors = nan(im_height, model.num_bins);
x_values = 1:im_height;

for ii = 1:model.num_bins
    
    this_sigma = model.stds(ii) * cropped_width;
    this_mean = model.means(ii) * cropped_width;
    
    this_prediction = 1 - normcdf(x_values, this_mean, this_sigma);
    
    bin_vectors(:, ii) = this_prediction;
    
end

%imagesc(bin_vectors);
bin_idxs = ceil(((1:cropped_width)/cropped_width)*model.num_bins);
output_image = zeros(im_height, original_width);
to_fill = find(~to_remove)

% now fill in output image
for ii = 1:length(to_fill)
    
    this_idx = to_fill(ii);
    
    this_bin = bin_idxs(ii); 
    
    temp_bin_vector = bin_vectors(1:end-depth(this_idx)+1, this_bin);
        
    output_image(depth(this_idx):end, this_idx) = temp_bin_vector;
        
end

% now add the outside rows to the image


% pad in the first row with ones
rendered_depth = fill_grid_from_depth(depth, im_height, 0);
output_image(rendered_depth == 1) = 1;

