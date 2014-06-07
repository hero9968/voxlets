% a script to do RGB segmentation of a region, given an initial
% segmentation from the depth image. This should become a function soon
% enough...

clear
cd ~/projects/shape_sharing/3D/src/
addpath(genpath('.'))
addpath(genpath('../../common/'))
addpath ../../2D/src/segment/
run ../define_params_3d.m

%% loading in some of the ECCV dataset
clear cloud
filepath = '/Users/Michael/data/others_data/ECCV_dataset/pcd_files/frame_20111220T111153.549117.pcd';
P = loadpcd(filepath);
cloud.xyz = P(:, :, 1:3);
cloud.xyz = reshape(permute(cloud.xyz, [3, 1, 2]), 3, [])';
cloud.rgb = P(:, :, 4:6);
cloud.depth = reshape(P(:, :, 3), [480, 640]);
[cloud.normals, cloud.curvature] = normals_wrapper(cloud.xyz, 'knn', 50);

%% running segment soup algorithm
[idxs, idxs_without_nans, probabilities, all_idx] = segment_soup_3d(cloud, params.segment_soup);

%% now extractin the region
use_hsv = 1;
gmm = 0;

segment_to_use = 8;
region.idx = reshape(idxs(:, segment_to_use), [480, 640]) > 0.5;
imagesc(region.idx)

% morpological operations
region.eroded = imerode(region.idx, strel('disk',3));
region.dilated1 = imdilate(region.idx, strel('disk',5));
region.dilated2 = imdilate(region.idx, strel('disk',25));
region.outer_mask = logical(region.dilated1 - region.dilated2);
imagesc(region.outer_mask)

%% extracting the pixels for inner and outer
if ~use_hsv
    cloud.col_reshape = reshape(permute(cloud.rgb, [3, 1, 2]), 3, [])';
else
    cloud.col_reshape = reshape(permute(rgb2hsv(double(cloud.rgb)), [3, 1, 2]), 3, [])';
    cloud.col_reshape = cloud.col_reshape(:, 1:3);
end
region.inner_pixels = cloud.col_reshape(region.eroded(:), :);
region.outer_pixels = cloud.col_reshape(region.outer_mask(:), :);

subplot(211); hist(region.inner_pixels, 100)
subplot(212); hist(region.outer_pixels, 100)

%% fitting colour models
if gmm
    region.inner_model = gmdistribution.fit(region.inner_pixels,4, 'Replicates', 3, 'SharedCov', false, 'CovType', 'diagonal');
    region.outer_model = gmdistribution.fit(region.outer_pixels,4, 'Replicates', 3, 'SharedCov', false, 'CovType', 'diagonal');
end   

%% now classifying all the pixels in the image, for visulaisaoitn purposes
if gmm
    region.inner_likelihood = pdf(region.inner_model, cloud.col_reshape);
    region.outer_likelihood = pdf(region.outer_model, cloud.col_reshape);
    region.inner_posterior = (region.inner_likelihood * 0.5) ./ ( region.inner_likelihood * 0.5 + region.outer_likelihood * 0.5);
else
    all_pixels = [region.inner_pixels; region.outer_pixels];
    pixel_labels = [0*region.inner_pixels(:, 1); 0*region.outer_pixels(:, 1)+1];
    [~, neighbours] = pdist2(all_pixels, cloud.col_reshape, 'euclidean', 'smallest', 20);
    labelling = pixel_labels(neighbours);
    region.inner_posterior = sum(labelling, 1) / 20;
    %region.outer_likelihood = pdf(region.outer_model, cloud.col_reshape);
end
region.outer_posterior = 1 - region.inner_posterior;
subplot(121)
imagesc(reshape(region.inner_posterior, [480, 640]))
axis image
subplot(122)
imagesc(reshape(region.outer_posterior, [480, 640]))
axis image

%% now set up the graphcut unary terms
region.outer_P = max(region.outer_posterior, region.outer_mask(:));
region.outer_P(region.eroded) = 0;
region.outer_P(~region.dilated2) = 1;
region.outer_P = reshape(region.outer_P, [480, 640]);
subplot(121)
imagesc(region.outer_P)
axis image
subplot(122)
imagesc(region.idx)
axis image

%% extract the bounding box...
margin = 10;
[im_height, im_width] = size(region.idx);
idx_start_X = find(any(region.idx, 1), 1, 'first');
idx_end_X = find(any(region.idx, 1), 1, 'last');
X_start = max(0, idx_start_X - margin);
X_end = min(im_width, idx_end_X + margin);
idx_start_Y = find(any(region.idx, 2), 1, 'first');
idx_end_Y = find(any(region.idx, 2), 1, 'last');
Y_start = max(0, idx_start_Y - margin);
Y_end = min(im_height, idx_end_Y + margin);
region.cropped = region.outer_P(Y_start:Y_end, X_start:X_end);
imagesc(region.cropped);
axis image
[t_gx, t_gy] = gradient(cloud.rgb);
region.gx_cropped = t_gx(Y_start:Y_end, X_start:X_end);
region.gy_cropped = t_gy(Y_start:Y_end, X_start:X_end);

%% running graphcut
datacost = -log(cat(3, region.cropped, 1-region.cropped));
datacost(isinf(datacost)) = 10;
labelcost = 100*(ones(2)-eye(2))
gch = GraphCut('open', datacost, labelcost, single(abs(region.gy_cropped)), single(abs(region.gx_cropped)));
[gch, labels] = GraphCut('expand', gch);
[gch] = GraphCut('close', gch)
imagesc(labels)
axis image

%%
%[X, Y] = find(edge(reshape(region.eroded(:), [480, 640])));
if 1
    [X, Y] = find(edge(labels==1));
    Y = Y+X_start-1;
    X = X+Y_start-1;
else
    [X, Y] = find(edge(region.idx));
end
imagesc(cloud.rgb)
hold on
plot(Y, X, '.')
hold off
axis image


%% alternate route
[edges, rgb_edges] = depth_edges(cloud.rgb, cloud.depth);
t_rgb = cloud.rgb;
t_rgb(:, :, 3) = (cloud.depth/max(cloud.depth(:))).^0.5;
t_rgb(:, :, 2) = (cloud.depth/max(cloud.depth(:))).^0.5;
subplot(131)
imagesc(t_rgb);
axis image
subplot(121)
imagesc(edges)
axis image
subplot(122)
imagesc(rgb_edges);
axis image
