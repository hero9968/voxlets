function plot_transforms(transformed, output_img, gt_img)

max_height = 200;

subplot(4, 5, 1); 
imagesc(gt_img(1:max_height, :))
axis image

for ii = 1:min(18, length(transformed)); 
    subplot(4, 5, ii+1); 
    imagesc(transformed(ii).masks(1:max_height, :)); 
    axis image
    hold on
    X = transformed(ii).transformed_depth(1, :);
    Y = transformed(ii).transformed_depth(2, :);
    plot(X, Y, 'g')
    hold off
end

subplot(4, 5, 20); 
imagesc(output_img(1:max_height, :))
axis image