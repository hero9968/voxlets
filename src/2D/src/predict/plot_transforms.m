function plot_transforms(transformed, output_img, gt_img)

%max_height = 200;

% settiping up subplot dimensions
plot_n = 7;
plot_m = 12;

% rendering the depth of the gt image
gt_depth = raytrace_2d(gt_img);
gt_y = 1:length(gt_depth);

% plotting the ground truth mask and the rendered depth
subaxis(plot_n, plot_m, 1, 'SpacingVert',0, 'SpacingHorizontal', 0); 
imagesc(gt_img)
hold on
plot(gt_y, gt_depth, 'c', 'LineWidth', 1.5)
hold off
axis image off
title('Input (blue) + GT image')

% deciding which basis shapes to use
to_use = 1:min(plot_n*plot_m - 2, length(transformed));

for jj = 1:length(to_use)
    
    ii = to_use(jj);
    
    % plotting the basis shape
    subaxis(plot_n, plot_m, jj+1); 
    imagesc(transformed(ii).cropped_mask); 
    colormap(flipgray)
    set(gca, 'clim', [0, 1])        
    axis image off
    
    % plotting the rendered depth from the basis shape
    hold on
    X = transformed(ii).transformed_depth(1, :) - transformed(ii).padding;
    Y = transformed(ii).transformed_depth(2, :) - transformed(ii).padding;
    plot(X, Y, 'g', 'LineWidth', 1.5)
    
    % plotting the input depth
    plot(gt_y, gt_depth, 'c', 'LineWidth', 1.5)
    
    title(['B_' num2str(jj)])
    hold off
end

% plotting the aggregated basis shapes
subaxis(plot_n, plot_m, plot_n*plot_m); 
imagesc(output_img)
axis image off
title('Combination of basis shapes')

% making the font larger for the whole figure
set(findall(gcf,'type','text'),'fontSize',14)