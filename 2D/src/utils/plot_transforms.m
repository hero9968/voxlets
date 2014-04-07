function plot_transforms(transformed, output_img, gt_img)

subplot(4, 5, 1); 
imagesc(gt_img(1:100, :))
axis image

for ii = 1:18; 
    subplot(4, 5, ii+1); 
    imagesc(transformed(ii).masks(1:100, :)); 
    axis image
end

subplot(4, 5, 20); 
imagesc(output_img(1:100, :))
axis image