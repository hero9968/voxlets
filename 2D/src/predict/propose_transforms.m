function transforms = propose_transforms(model, depth, params)

outlier_distance = params.icp.outlier_distance;
number_matches_to_use = params.num_proposals;
num_samples = params.shape_dist.num_samples;

% compute shape distribution for the depth
model_XY = [1:length(depth); double(depth)];
model_XY(:, any(isnan(model_XY), 1)) = [];

% choosing a scale for the XY points
if model.scale_invariant
    scale = normalise_scale(model_XY);
else
    scale = 1;
end

% computing the feature vector for the depth image
model_XY_resized = scale * model_XY;

if model.sd_angles
    angle_edges = model.angle_edges;
    norms = normals_radius_2d(model_XY, params.normal_radius);
    shape_dist = shape_distribution_2d_angles(model_XY_resized, norms, num_samples, model.bin_edges, angle_edges);
else
    shape_dist = shape_distribution_2d(model_XY_resized, num_samples, model.bin_edges);
end

% find top matching shape distribution(s) by chi-squared distance
all_dists = cell2mat(model.shape_dists)';
dists = pdist2(shape_dist', all_dists, 'chisq');
[~, idx] = sort(dists, 'ascend');

% now align in the match using PCA
[~, ~, model_transform_from_origin] = transformation_to_origin_2d(model_XY);

% initialising empty transforms vector
transforms = [];
% .data_idx  scalar
% .pca_transform (3x3)
% .icp_transform (3x3)
% .flipped   binary
count = 1;

% form the complete rotation from each of the top matches to the shape
for ii = 1:number_matches_to_use
    
    this_idx = idx(ii);
    
    % getting a matrix to scale the matched depth image
    scale_m = scale_matrix(model.scales(this_idx) / scale);
    
    % find transformaiton from top match to this object...
    flip_m{1} = [1, 0, 0; 0 -1 0; 0 0 1];
    flip_m{2} = eye(3);
        
    % loop over not flipped/flipped
    for jj = 1:length(flip_m)
        
        % stack all transforms together
        transforms(count).pca = model_transform_from_origin * scale_m * flip_m{jj} * inv(model.transf{this_idx});
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
        %transforms(count).image = transform_image(transforms(count).base_image, this_img_transform);
        transforms(count).ii = ii;
        transforms(count).depth = data_depth;

        count = count + 1;
    end
end
    
    