% script to propose 3D basis shapes for a specific region, and to somehow
% visualise. No transformations or anything clever like that yet...

% a script to load in a depth image, convert to xyz, compute normals and segment
clear
cd ~/projects/shape_sharing/src/3D/src/
addpath(genpath('.'))
addpath(genpath('../../2D/src'))
addpath(genpath('../../common/'))
run ../define_params_3d
load(paths.structured_model_file, 'model')

%% loading in some of the ECCV dataset
cloud = loadpgm_as_cloud('~/projects/shape_sharing/data/3D/scenes/first_few_render_noisy00000.pgm', params.full_intrinsics);
[cloud.normals, cloud.curvature] = normals_wrapper(cloud.xyz, 'knn', 50);

%% running segment soup algorithm
[idxs, idxs_without_nans, probabilities, all_idx] = segment_soup_3d(cloud, params.segment_soup);

%% plotting segments
plot_segment_soup_3d(cloud.rgb.^0.2, idxs, probabilities);

%% Choosing a segment and computing the feature vector
seg_index = 12;
segment = extract_segment(cloud, idxs(:, seg_index), params);
matches = propose_matches(segment, model, 20, 'shape_dist', params, paths);

%% plotting the closest matches
plot_matches(matches, 20, segment.mask, params, paths)

%% now must align the match into the original image
imagesc(cloud.depth)
hold on
plot(segment.centroid(1), segment.centroid(2), 'r+')
hold off

%%
clf
T = cloud.depth;
for ii = 1:10
    
    trans1 = translation_matrix(-matches(ii).centroid(1), -matches(ii).centroid(2));
    rot = rotation_matrix(matches(ii).angle);
    trans2 = translation_matrix(segment.centroid(1), segment.centroid(2));
    scale = scale_matrix(segment.scale / matches(ii).scale);
    
    transf = maketform('affine', (trans2 * rot * trans1)');
    [H, W] = size(cloud.depth);
    
    depth = matches(ii).depth;
    depth(isnan(depth)) = 0;
    translated_match = imtransform(depth, transf, 'nearest', 'XData', [1 W], 'YData', [1 H], 'size', size(T));
    T = translated_match + T;
    
end

imagesc(T)








