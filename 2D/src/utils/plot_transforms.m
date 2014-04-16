function plot_transforms(transformed, output_img, gt_img)

max_height = 200;

gt_depth = raytrace_2d(gt_img);
gt_y = 1:length(gt_depth);

subplot(4, 5, 1); 
imagesc(gt_img(1:max_height, :))
axis image

for ii = 1:min(18, length(transformed)); 
    subplot(4, 5, ii+1); 
    imagesc(transformed(ii).extended_mask(1:max_height, :)); 
    axis image
    hold on
    X = transformed(ii).transformed_depth(1, :) - transformed(ii).padding;
    Y = transformed(ii).transformed_depth(2, :) - transformed(ii).padding;
    plot(X, Y, 'g')
    plot(gt_y, gt_depth, 'c')
    hold off
end

subplot(4, 5, 20); 
imagesc(output_img(1:max_height, :))
axis image