function transforms = propose_transforms(model, depth, norms, params)

outlier_distance = params.icp.outlier_distance;
number_matches_to_use = params.num_proposals;
num_samples = params.shape_dist.num_samples;

% compute shape distribution for the depth
model_XY = [1:length(depth); double(depth)];
to_remove = any(isnan(model_XY), 1);
model_XY(:, to_remove) = [];
norms(:, to_remove) = [];

% choosing a scale for the XY points
if model.scale_invariant
    scale = normalise_scale(model_XY);
else
    scale = 1;
end

% computing the feature vector for the depth image
model_XY_scaled = scale * model_XY;
shape_dist = ...
        shape_dist_2d_dict(model_XY_scaled, norms, num_samples, model.dist_angle_dict);

% find top matching shape distribution(s) by chi-squared distance
all_dists = cell2mat({model.training_data.shape_dist}');
dists = pdist2a(shape_dist, all_dists, 'chisq');
%dists = pdist2(shape_dist, all_dists, 'euclidean');
assert(size(dists, 2)==size(all_dists, 1));
assert(size(dists, 1)==1);

[dists_sorted, idx] = sort(dists, 'ascend');

% now align in the match using PCA
[~, ~, model_T_from_origin] = transformation_to_origin_2d(model_XY);

% initialising empty transforms vector
transforms = [];
count = 1;

% form the complete rotation from each of the top matches to the shape
for ii = 1:number_matches_to_use
    
    this_idx = idx(ii);
    
    % getting a matrix to scale the matched depth image
    scale_m = scale_matrix(model.training_data(this_idx).scale / scale);
    
    % find transformaiton from top match to this object...
    flip_m{1} = [1, 0, 0; 0 -1 0; 0 0 1];
    flip_m{2} = eye(3);
        
    % loop over not flipped/flipped
    for jj = 1:length(flip_m)
        
        % stack all transforms together
        data_T_from_origin = model.training_data(this_idx).transform_to_origin;
        transforms(count).pca = model_T_from_origin * scale_m * flip_m{jj} * inv(data_T_from_origin);
        check_isgood_transform( transforms(count).pca )
        
        % rotate the data depth to initial guess
        data_depth = model.training_data(this_idx).depth;
        data_XY = [1:length(data_depth);  double(data_depth)];
        
        % calling icp routine. Some more params are set inside the wrapper function        
        transforms(count).icp = icp_wrapper(model_XY, data_XY, transforms(count).pca, outlier_distance);

        transforms(count).flipped = jj==1;
        transforms(count).data_idx = this_idx;
        transforms(count).image_idx = model.training_data(this_idx).image_idx;
        transforms(count).base_image = model.images{transforms(count).image_idx};
        transforms(count).img_transform = model.training_data(this_idx).transform; 
        transforms(count).ii = ii;
        transforms(count).chi_squared = dists_sorted(this_idx);
        transforms(count).depth = data_depth;
        transforms(count).scale = model.training_data(this_idx).scale;
        
        % add this transform into the main vector
        count = count + 1;
    end
end
    
    