function transforms = propose_transforms(data, depth, params)

outlier_distance = 10;%params.icp.outlier_distance;
number_matches_to_use = params.num_proposals;


% compute shape distribution for the depth
model_XY = [1:length(depth); double(depth)];
model_XY(:, any(isnan(model_XY), 1)) = [];

if data.scale_invariant
    bin_edges = data.bin_edges;
    scale = normalise_scale(model_XY);
else
    bin_edges = data.bin_edges;
    scale = 1;
end

num_samples = params.shape_dist.num_samples;

tX = scale * model_XY(1, :)';
tY = scale * model_XY(2, :)';

if data.sd_angles
    angle_edges = data.angle_edges;
    norms = normals_radius_2d(model_XY, params.normal_radius);
    shape_dist = shape_distribution_2d_angles([tX'; tY'], norms, num_samples, bin_edges, angle_edges);
else
    shape_dist = shape_distribution_2d(tX, tY, num_samples, bin_edges);
end

% find top matching shape distribution(s) by chi-squared distance
all_dists = cell2mat(data.shape_dists)';
%dists = pdist2(shape_dist', all_dists, 'chisq');
dists = pdist2(shape_dist', all_dists, 'chisq');
[~, idx] = sort(dists, 'ascend');

% now align in the match using PCA
[~, ~, model_transform_from_origin] = transformation_to_origin_2d(model_XY(1, :), model_XY(2, :));

% 
transforms = [];
% .data_idx  scalar
% .pca_transform (3x3)
% .icp_transform (3x3)
% .flipped   binary
count = 1;

% form the complete rotation from each of the top matches to the shape
for ii = 1:number_matches_to_use
    
    this_idx = idx(ii);
    %this_scale = scale / data.scales(this_idx);
    this_scale = data.scales(this_idx) / scale;
    scale_m = [this_scale, 0, 0; ...
              0, this_scale, 0; ...
              0 0 1];
    
    % find transformaiton from top match to this object...
    flip_m{1} = [1, 0, 0; 0 -1 0; 0 0 1];
    flip_m{2} = eye(3);
        
    % loop over not flipped/flipped
    for jj = 1:length(flip_m)
        
        transforms(count).pca = model_transform_from_origin * scale_m * flip_m{jj} * inv(data.transf{this_idx});

        if cond(transforms(count).pca) > 1e8
            disp(['Test - Seems like conditioning is bad'])
            keyboard
        end
        
        % rotate the data depth to initial guess
        data_depth = data.depths{this_idx};
        data_XY = [1:length(data_depth);  double(data_depth)];
        
        % calling icp routine. Some more params are set inside the wrapper function        
        transforms(count).icp = icp_wrapper(model_XY, data_XY, transforms(count).pca, outlier_distance);

        transforms(count).flipped = jj==1;
        transforms(count).data_idx = this_idx;
        transforms(count).image = data.images{this_idx};
        transforms(count).ii = ii;
        transforms(count).depth = data.depths{this_idx};

        count = count + 1;
    end
end
    
    