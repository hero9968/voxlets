function [stacked_image, transforms] = test_fitting_model(model, depth, im_height, params)
% image = test_fitting_model(model, depth, params)
%
% applys the learned model to predict the image that might be present which
% gave the depth. The parameters in params might eventually be moved to
% model.

% input checks
assert(isvector(depth));

% get vector structure of possible training images which match the depth, 
% and their transformation into the scene
transforms = propose_transforms(model, depth, params);

% aggregating the possible transforms into an output image
if params.aggregating
    [~, stacked_image] = aggregate_masks(transforms, im_height, depth, params);
else
    stacked_image = [];
end




%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Everything below this line is just about plotting...
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

plotting_transforms = params.plotting.plot_transforms;
plotting_matches = params.plotting.plot_matches;

if plotting_transforms
    num_to_plot = 18;
    model_XY = [1:length(depth); double(depth)];
    plot_transforms(transforms, model, model_XY, num_to_plot);
end

if plotting_matches

    %axis image
    title('Model data');
    [n, m] = best_subplot_dims(params.plotting.num_matches);
    
    for ii = 1:params.plotting.num_matches
                
        subplot(n, m, ii)
        
        % extracting the data edge image
        this_idx = transforms(ii).data_idx;
        this_image = model.images{this_idx};
        this_depth = model.depths{this_idx};
        
        % plot the model depth image
        combine_mask_and_depth(this_image, this_depth);

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


end



function plot_transforms(transforms, model, model_XY, num_to_plot)

% plot different trasnforms on top of the current mask
cols = {'r','g', 'b', 'k', 'c', 'y', 'r','g', 'b', 'k', 'c', 'y', 'r','g', 'b', 'k', 'c', 'y'};

X = model_XY(1, :);
Y = model_XY(2, :);

for ii = 1:min(num_to_plot, length(transforms))

    % extracting the data edge image
    this_idx = transforms(ii).data_idx;
    image_idx = transforms(ii).image_idx;
    this_image = model.images{image_idx};
    %[Yf, Xf] = find(edge(this_image));
    Yf = model.training_data(this_idx).depth';
    Xf = (1:length(Yf))';

    % plot the model depth image
    subplot(1, 4, 1);        hold on
    plot(X, Y, cols{ii}, 'linewidth', 3);

    hold off; axis image
    if transforms(ii).flipped; marker = 'o'; else marker = '^'; end

    % plot the pca transform
    XYf_rot = apply_transformation_2d([Xf'; Yf'], transforms(ii).pca);
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
