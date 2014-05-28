function plot_segment_soup( binary_segmentation, depth, titles )
% plots a subplot for each item in the segment soup

[n, m] = best_subplot_dims(size(binary_segmentation, 1));

Y = 1:length(depth);

% plot each segmentation in a separate subplot
for jj = 1:size(binary_segmentation, 1)
    
    inliers = binary_segmentation(jj, :);
    
    subplot(n, m, jj);
    
    % plotting the items not in this segmentation
    plot(Y(~inliers), depth(~inliers), 'or', ...
        'MarkerFaceColor', [0.5, 0.5, 0.5],...
        'MarkerEdgeColor', 'none', ...
        'MarkerSize', 4)

    hold on
    
    % plotting the items in this segmentation
    plot(Y(inliers), depth(inliers), 'o', ...
        'MarkerFaceColor', [1, 0.1, 0.1],...
        'MarkerEdgeColor', 'none', ...
        'MarkerSize', 4)
    
    hold off
    
    set(gca,'YDir','reverse');
    axis image
    
    if nargin == 3
        title(titles{jj})
    end
    
    
end