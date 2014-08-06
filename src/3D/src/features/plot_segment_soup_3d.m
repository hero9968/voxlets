function plot_segment_soup_3d(base_image, segmentation, extra_titles)
% function to plot the results of a segment soup algorithm

[h, w, ~] = size(base_image);
assert(size(segmentation, 1) == h*w);

[n, m] = best_subplot_dims(size(segmentation, 2));

% plotting each segmentation on separate subplot
for ii = 1:size(segmentation, 2)
    
    temp_image = reshape(segmentation(:, ii), h, w) > 0.5;
    
    subplot(n, m, ii)
    plot_depth_segmentation(base_image.^0.2, temp_image);
    
    if nargin == 3
        title([num2str(ii), ' - ', num2str(extra_titles(ii))])
    else
        title(num2str(ii))
    end
end

set(findall(gcf,'type','text'),'fontSize',18,'fontWeight','bold')