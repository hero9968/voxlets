% script to propose 3D basis shapes for a specific region, and to somehow
% visualise. No transformations or anything clever like that yet...

% a script to load in a depth image, convert to xyz, compute normals and segment
clear
cd ~/projects/shape_sharing/3D/src/
addpath plotting/
addpath features/
addpath ./file_io/matpcl/
addpath ../../common/
addpath transformations/
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

%% plotting segments
close all
plot_segment_soup_3d(cloud.rgb.^0.2, idxs);
for ii = 1:length(probabilities)
    subplot(2, 4, ii)
    title(num2str(probabilities(ii)))
end
set(findall(gcf,'type','text'),'fontSize',18,'fontWeight','bold')

%% Choosing a segment and computing the feature vector
segment.seg_index = 5;
segment.idx = idxs(:, segment.seg_index);
segment.xyz = cloud.xyz(segment.idx>0.5, :);
segment.scaled_xyz = segment.xyz * normalise_scale(segment.xyz);
