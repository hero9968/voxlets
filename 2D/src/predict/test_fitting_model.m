function stacked_image = test_fitting_model(data, depth, params)
% image = test_fitting_model(model, depth, params)
%
% applys the learned model to predict the image that might be present which
% gave the depth. The parameters in params might eventually be moved to
% model.
plotting = 1;
plotting_matches = 0;
outlier_distance = 10;%params.icp.outlier_distance;
number_matches_to_use = 1;

% input checks
assert(isvector(depth));

% compute shape distribution for the depth
model_XY = [1:length(depth); double(depth)];
model_XY(:, any(isnan(model_XY), 1)) = [];

if data.scale_invariant
    bin_edges = params.shape_dist.si_bin_edges;
    scale = normalise_scale(model_XY);
else
    bin_edges = params.shape_dist.bin_edges;
    scale = 1;
end

num_samples = params.shape_dist.num_samples;

tX = scale * model_XY(1, :)';
tY = scale * model_XY(2, :)';

shape_dist = shape_distribution_2d(tX, tY, num_samples, bin_edges);

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

        if cond(transforms(count).pca) > 1e7
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
    
    
% plot different trasnforms on top of the current mask
cols = {'r','g', 'b', 'k', 'c', 'y', 'r','g', 'b', 'k', 'c', 'y', 'r','g', 'b', 'k', 'c', 'y'};
if plotting
    X = model_XY(1, :);
    Y = model_XY(2, :);
    
    for ii = 1:min(18, length(transforms))
        
        % extracting the data edge image
        this_idx = transforms(ii).data_idx;
        this_image = data.images{this_idx};
        %[Yf, Xf] = find(edge(this_image));
        Yf = data.depths{this_idx}';
        Xf = (1:length(Yf))';
        
        % plot the model depth image
        subplot(1, 4, 1);        hold on
        plot(X, Y, cols{ii}, 'linewidth', 3);
        
        hold off; axis image
        if transforms(ii).flipped; marker = 'o'; else marker = '^'; end
        
        % plot the pca transform
        XYf_rot = apply_transformation_2d([Xf'; Yf'], transforms(ii).pca)
        subplot(1, 4, 2);        hold on
        colour_string = [cols{transforms(ii).ii}];
        plot(XYf_rot(1, :), XYf_rot(2, :), marker, 'markersize', 2+1*(+transforms(ii).flipped), 'markerfacecolor', colour_string, 'markeredgecolor', 'none');
        hold off; axis image
        
        % plot the icp transform
        XYf_rot = apply_transformation_2d([Xf'; Yf'], transforms(ii).icp);
        subplot(1, 4, 3);        hold on
        colour_string = [cols{transforms(ii).ii}];
        plot(XYf_rot(1, :), XYf_rot(2, :), marker, 'markersize', 2+1*(+transforms(ii).flipped), 'markerfacecolor', colour_string, 'markeredgecolor', 'none');
        hold off; axis image
    end
end


if params.aggregating
    
    % now creating the combined image
    prediction_masks = {transforms.image};
    prediction_transformations = {transforms.icp};
    prediction_weights = ones(1, length(transforms)) / length(transforms);
    [stacked_image_full, x_data, y_data] = ...
        aggregate_depth_predictions(prediction_masks, prediction_transformations, prediction_weights);

    % cropping and resizing the image
    x_range = x_data > 0 & x_data <= length(depth);
    y_range = y_data > 0 & y_data <= params.im_height;
    stacked_image_cropped = stacked_image_full(y_range, x_range);
    if isempty(stacked_image_cropped)
        stacked_image_cropped = zeros(params.im_height, length(depth));
    else
        extra_height = params.im_height - size(stacked_image_cropped, 1);
        stacked_image_cropped = [stacked_image_cropped; zeros(extra_height, length(depth))];
    end

    % applying the known mask for the known free pixels
    known_mask = fill_grid_from_depth(depth, params.im_height, 0.5);
    stacked_image_cropped(known_mask==0) = 0;
    stacked_image_cropped(known_mask==1) = 1;


    stacked_image = stacked_image_cropped;
else
    stacked_image = [];
end

if plotting_matches

    %subplot(4, 4, 1);
    %plot(model_XY(1, :), model_XY(2, :), 'r');
    
    %axis image
    title('Model data');
    
    for ii = 1:4%length(transforms)
        subplot(3, 4, 4+ii + 4 * data.scale_invariant)
        
        % extracting the data edge image
        this_idx = transforms(ii).data_idx;
        this_image = data.images{this_idx};
        this_depth = data.depths{this_idx};
        
        % plot the model depth image
        combine_mask_and_depth(this_image(1:100, :), this_depth);

    end
end


if 0
    axis image
    for ii = 1:5
        this_idx = idx(ii);

        % creating combined depth and mask image
        subplot(1, 6, ii+1);
        combine_mask_and_depth(model.images{this_idx}, model.depths{this_idx});
        title(num2str(dists(this_idx)));
    end
end

