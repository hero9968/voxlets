function image = test_fitting_model(model, depth, params)
% image = test_fitting_model(model, depth, params)
%
% applys the learned model to predict the image that might be present which
% gave the depth. The parameters in params might eventually be moved to
% model.

% input checks
assert(isvector(depth));

% set up params etc.
num_samples = params.shape_dist.num_samples;
bin_edges = params.shape_dist.bin_edges;
number_matches_to_use = 10;
cols = {'r','g', 'b', 'k', 'c', 'y', 'r','g', 'b', 'k', 'c', 'y', 'r-','g:', 'b--', 'k', 'c', 'y'};

% compute shape distribution
Y = (double(depth));
X = 1:length(Y);
shape_dist = shape_distribution_2d(X(:), Y(:), num_samples, bin_edges);

% find top matching shape distribution(s) by chi-squared distance
%chisq = @(xi, yi)( sum( (xi-yi).^2 / (xi+yi) ) / 2 );
all_dists = cell2mat(model.shape_dists)';
dists = pdist2(shape_dist', all_dists, 'chisq');%, chisq);
[~, idx] = sort(dists, 'ascend');

% now align in the match using PCA
[~, ~, this_transform_to_origin] = transformation_to_origin_2d(X, Y);

% form the complete rotation from each of the top matches to the shape
for ii = 1:number_matches_to_use
    
    this_idx = idx(ii);
    
    % find transformaiton from top match to this object...
    flip_m = [1, 0, 0; 0 -1 0; 0 0 1];
    
    full_transformation = this_transform_to_origin * inv(model.transf{this_idx});
    full_transformation_flipped = this_transform_to_origin * flip_m * inv(model.transf{this_idx});
    
    % now rotate this other image
    other_depth = model.depths{this_idx};
    X = 1:length(other_depth);
    Y = double(other_depth);
    
    XY = [X; Y; ones(1, length(other_depth))];
    XY_rot = apply_transformation_2d([X; Y], full_transformation);
    XY_rot_flipped = apply_transformation_2d([X; Y], full_transformation_flipped);

    % now do for the full masks!
    this_image = model.images{this_idx};
    [Yf, Xf] = find(edge(this_image));
    XYf_rot = apply_transformation_2d([Xf'; Yf'], full_transformation);
    XYf_rot_flipped = apply_transformation_2d([Xf'; Yf'], full_transformation_flipped);

    % plot on top of the current mask
    subplot(1, 3, 1)
    hold on
    plot(X, Y, cols{ii}, 'linewidth', 3);
    

    plot_full = true;

    subplot(1, 3, 2);
    hold on
    if plot_full
        plot(XYf_rot(1, :), XYf_rot(2, :), [cols{ii} '.'], 'markersize', 7);
    else
        plot(XY_rot(1, :), XY_rot(2, :), cols{ii}, 'linewidth', 3);
    end

    subplot(1, 3, 3);
    hold on
    if plot_full
        plot(XYf_rot_flipped(1, :), XYf_rot_flipped(2, :), [cols{ii} '.'], 'markersize', 8);
    else
        plot(XY_rot_flipped(1, :), XY_rot_flipped(2, :), cols{ii}, 'linewidth', 3);
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

