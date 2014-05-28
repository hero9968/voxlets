% propose transforms for an image and plot them nicely

cd ~/projects/shape_sharing/2D/src
clear
run ../define_params
addpath predict
addpath utils
addpath transformations/
addpath(genpath('external'))
addpath ../../common/

%% loading in model and test data
load(paths.test_data, 'test_data')
load(paths.all_images, 'all_images')
load(paths.structured_predict_si_model_path, 'model');

%% loading in the data
num = 4000;
this_img = test_data(num);
this_img.raw_image = all_images{this_img.image_idx};
this_img.gt_image = rotate_image_nicely(this_img.raw_image, this_img.angle);

%% diplaying the raw img rotated to different angles
close all
angs = linspace(0, 360, 100);
for ii = 1:length(angs)
    temp_rotated_img = rotate_image_nicely(this_img.raw_image, angs(ii));
    subplot(10, 10, ii);
    imagesc(temp_rotated_img);
    axis image
end

%% setting parameters for the transformation proposals
params.num_proposals = 2000;
params.apply_known_mask = 0;
params.transform_type = 'icp';
params.icp.outlier_distance = 50;

%% propose transforms and aggregate
%profile on
%transforms = propose_transforms(model, depth, params);
transforms = ...
    propose_segmented_transforms(model, this_img.depth, this_img.normals, this_img.segmented, params);
%profile off viewer
%%
%transforms2 = transforms(randperm(length(transforms)));
[out_img, out_img_cropped, transformed] = ...
    aggregate_masks(transforms, size(this_img.gt_image, 1), this_img.depth, params);

%% now finding some metrics to assess the goodness of fit
imheight = 500;
gt_depth = raytrace_2d(this_img.gt_image);
%filled_from_depth = fill_grid_from_depth(gt_depth, imheight, 1);
filled_trans = nan(imheight, length(gt_depth));
for jj = 1:length(gt_depth)
    this_depth = gt_depth(jj);
    filled_trans(:, jj) = normpdf(1:500, this_depth, 10);
end
imagesc(filled_trans)
axis image
colorbar

% getting the ground truth points
gtX = 1:length(gt_depth);
gtY = gt_depth;
to_remove = isnan(gt_depth);
gtX = gtX(~to_remove);
gtY = gtY(~to_remove);

%%
for ii = 1:length(transformed)
    
    % this prediction position
    X = transformed(ii).transformed_depth(1, :) - transformed(ii).padding;
    Y = transformed(ii).transformed_depth(2, :) - transformed(ii).padding;
    
    % getting the predicted points
    to_remove = isnan(X) | isnan(Y) | X < 1 | X > size(filled_trans, 2) | Y < 1 | Y > size(filled_trans, 1);
    rX = (X(~to_remove));
    rY = (Y(~to_remove));
    
    % doing some kind of distance matrix
    T = pdist2([gtX(:), gtY(:)], [rX(:), rY(:)]);
    dists = min(T, [], 2);
    % make robust
    %dists(dists>10) = 10;
    transformed(ii).dist_to_gt = sum(dists.^2) / length(dists);
    
    if isempty(transformed(ii).dist_to_gt)
        transformed(ii).dist_to_gt = inf;
    end
    
    
    rInd = sub2ind(size(filled_trans), round(rY), round(rX));
    T2 = filled_trans(rInd);
    transformed(ii).inlier_sum = nansum(T2) / length(rX);
    if nansum(T2) == 0
        transformed(ii).inlier_sum = 0;
    end
        
    
    % plotting
    if 0
        clf
        imagesc(filled_trans)
        hold on
        plot(X, Y);
        plot(gtX, gtY, 'r')
        hold off
        axis image
    end
end

%%
[~, idx] = sort([transformed.dist_to_gt], 'ascend');
transformed2 = transformed(idx);
%plot([transformed2.dist_to_gt])
%plot(dists(idx))
plot_transforms(transformed2, out_img_cropped, this_img.gt_image);

%% reweighting the final image using the found weightings
probs =  exp(-[transformed2.dist_to_gt]);
probs = probs / sum(probs);
mask_stack = cell2mat(reshape({transformed2.cropped_mask}, 1, 1, []));
[~, weighted_stack] = noisy_or(mask_stack, 3, probs);
summed_stack = sum(weighted_stack, 3);
imagesc2(summed_stack)




