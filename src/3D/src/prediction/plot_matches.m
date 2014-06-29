function plot_matches(matches, num_to_plot, segment_mask, params, paths)
% plot the 3D matches which have been found...

[p, q] = best_subplot_dims(num_to_plot);

subplot(p, q, 1)
imagesc2(boxcrop_2d(segment_mask))
set(gcf, 'color', [1, 1, 1])

for ii = 1:(num_to_plot-1)
    
    this.model = params.model_filelist{matches(ii).model_idx};
    this.path = sprintf(paths.basis_models.rendered, this.model, matches(ii).view);
    
    % load and rotate the depth image
    load(this.path, 'depth')
    depth = format_depth(depth);
    t_depth = imrotate(depth, -matches(ii).angle);
    t_depth = boxcrop_2d(t_depth);
    
    % plot the depth image
    subplot(p, q, ii+1)
    plot_depth(t_depth)
    title([num2str(matches(ii).model_idx) ' - ' num2str(matches(ii).view)])
    
end