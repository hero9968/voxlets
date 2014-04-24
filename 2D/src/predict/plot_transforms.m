function plot_transforms(transformed, output_img, gt_img)

max_height = 200;

% settiping up subplot dimensions
plot_n = 2;
plot_m = 5;

% rendering the depth of the gt image
gt_depth = raytrace_2d(gt_img);
gt_y = 1:length(gt_depth);

% plotting the ground truth mask and the rendered depth
subplot(plot_n, plot_m, 1); 
imagesc(gt_img(1:max_height, :))
hold on
plot(gt_y, gt_depth, 'c', 'LineWidth', 1.5)
hold off
axis image off
title('Input (blue) + GT image')

% deciding which basis shapes to use
to_use = 1:min(8, length(transformed));

for jj = 1:length(to_use)
    
    ii = to_use(jj);
    
    % plotting the basis shape
    subplot(plot_n, plot_m, jj+1); 
    imagesc(transformed(ii).cropped_mask(1:max_height, :)); 
    colormap(flipgray)
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
subplot(plot_n, plot_m, 10); 
imagesc(output_img(1:max_height, :))
axis image off
title('Combination of basis shapes')

% making the font larger for the whole figure
set(findall(gcf,'type','text'),'fontSize',14)