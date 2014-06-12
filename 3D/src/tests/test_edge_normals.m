
clear
cd ~/projects/shape_sharing/3D/src/tests
addpath(genpath('.'))
addpath(genpath('../../../common/'))
addpath ../../2D/src/segment/
addpath ../features/
addpath ../../../2D/src/utils/
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

%% extracint the segment and computing the edge normals
close all
clear H
for to_use = 1:8;
    
    this.idx = idxs(:, to_use);
    this.mask = reshape(this.idx, size(cloud.depth)) > 0;
    %this.mask = zeros(100)
    %this.mask(20:80, 20:80) = 1;
    %imagesc(this.mask)
    
    [XY, norms] = edge_normals(this.mask, 30);
    %plot_edge_normals(XY', norms')
    
    XY = XY *  normalise_scale_2d([XY']);

    %
    [fv, dists_original, d_orig] = shape_dist_2d_dict(XY', norms', 20000, []);
    d_orig = max(min(d_orig, 1), -1);
    angles_original = acos(d_orig);
    H{to_use} = hist2d([dists_original(:), angles_original(:)], linspace(0, 1.2, 20), linspace(0, pi, 20));
    subplot(4, 2, to_use)
    imagesc(H{to_use})
    %plot(angles_original(:), dists_original(:), '.')
    axis image

end

%%
subplot(222)
barh(flipud(histc(angles_original(:), linspace(-1, 1, 30))))
subplot(223)
bar(sum(H, 1))

%plot(dists_original, angles_original, 'o')



%% computing some kind of FV for the edges and normals
temp = regionprops(this.mask, 'Centroid');
C = temp.Centroid;
hold on
plot(C(2), C(1), '+')
hold off

dists = XY - repmat(C, size(XY, 1), 1);
F1 = sqrt(dists(:, 1).^2 + dists(:, 2).^2)
dists_norm = normalise_length(dists);
F2 = dot(dists_norm, norms, 2);
plot(F1, F2, 'o')
%hist(F2, 20)

%%
hist(F1, 20)

for ii = 1:size(XY, 1)
    
    
    
end


