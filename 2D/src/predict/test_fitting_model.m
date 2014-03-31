function image = test_fitting_model(data, depth, params)
% image = test_fitting_model(model, depth, params)
%
% applys the learned model to predict the image that might be present which
% gave the depth. The parameters in params might eventually be moved to
% model.

plotting = true;
plot_full = true;
do_icp = true;
outlier_distance = 10;%params.icp.outlier_distance;
number_matches_to_use = 5;

% input checks
assert(isvector(depth));

% set up params etc.
num_samples = params.shape_dist.num_samples;
bin_edges = params.shape_dist.bin_edges;

cols = {'r','g', 'b', 'k', 'c', 'y', 'r','g', 'b', 'k', 'c', 'y', 'r-','g:', 'b--', 'k', 'c', 'y'};

% compute shape distribution
Y = (double(depth));
X = 1:length(Y);
model_XY = [X; Y];
shape_dist = shape_distribution_2d(X(:), Y(:), num_samples, bin_edges);

% find top matching shape distribution(s) by chi-squared distance
%chisq = @(xi, yi)( sum( (xi-yi).^2 / (xi+yi) ) / 2 );
all_dists = cell2mat(data.shape_dists)';
dists = pdist2(shape_dist', all_dists, 'chisq');%, chisq);
[~, idx] = sort(dists, 'ascend');

% now align in the match using PCA
[~, ~, model_transform_to_origin] = transformation_to_origin_2d(X, Y);

% form the complete rotation from each of the top matches to the shape
for ii = 1:number_matches_to_use
    
    this_idx = idx(ii);
    
    % find transformaiton from top match to this object...
    flip_m = [1, 0, 0; 0 -1 0; 0 0 1];
    
    full_pca_transform{1} = model_transform_to_origin * inv(data.transf{this_idx});
    full_pca_transform{2} = model_transform_to_origin * flip_m * inv(data.transf{this_idx});
    
    % loop over not flipped/flipped
    for jj = 1:2
        
        % rotate the data depth to initial guess
        other_depth = data.depths{this_idx};
        data_XY = [1:length(other_depth);  double(other_depth)];
        
        %XY = [X; Y; ones(1, length(other_depth))];
        %XY_rot = apply_transformation_2d([X; Y], full_pca_transform);
        %XY_rot_flipped = apply_transformation_2d([X; Y], full_pca_transform_flipped);

        % performing ICP to refine alignment
        icp_transform{ii, jj} = icpMex(model_XY, data_XY, full_pca_transform{jj}, outlier_distance, 'point_to_plane');
        %XY_rot2 = apply_transformation_2d(data_XY, icp_transform);

        % now do for the full masks!
        this_image = data.images{this_idx};
        [Yf, Xf] = find(edge(this_image));
        XYf_rot = apply_transformation_2d([Xf'; Yf'], full_pca_transform{jj});
        XYf_rot_flipped = apply_transformation_2d([Xf'; Yf'], icp_transform{ii, jj});
    end
 
    
    % plot on top of the current mask
    if plotting
        for jj = 1:2
            subplot(2, 3, 1+(jj-1)*3)
            hold on
            plot(X, Y, cols{ii}, 'linewidth', 3);

            subplot(2, 3, 2+(jj-1)*3);
            hold on
            if plot_full
                plot(XYf_rot(1, :), XYf_rot(2, :), [cols{ii} '.'], 'markersize', 7);
            else
                plot(XY_rot(1, :), XY_rot(2, :), cols{ii}, 'linewidth', 3);
            end

            subplot(2, 3, 3+(jj-1)*3);
            hold on
            if plot_full
                plot(XYf_rot_flipped(1, :), XYf_rot_flipped(2, :), [cols{ii} '.'], 'markersize', 8);
            else
                plot(XY_rot_flipped(1, :), XY_rot_flipped(2, :), cols{ii}, 'linewidth', 3);
            end
        end
    end
end



%%



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

