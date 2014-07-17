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
seg_index = 10;
segment = extract_segment(cloud, idxs(:, seg_index), params);
matches = propose_matches(segment, model, 20, 'shape_dist', params, paths);

%% plotting the closest matches
%plot_matches(matches, 20, segment.mask, params, paths)

%% 3D alignment visualisation on a per-region basis
plot3d(cloud.xyz);
for ii = 1:20
    
    % translating the basis shape to the origin
    trans1 = translation_matrix_3d(-matches(ii).centroid_3d);
    rot1 = (transformation_matrix_from_vector(matches(ii).centroid_normal, 1));
    
    % resolving the rotation in the camera plane
    camera_rot = rotation_matrix(matches(ii).angle + 1.5*7.2);
    camera_rot = [1, 0, 0, 0; zeros(3, 1), camera_rot];
    
    % translation from the origin to the scene segment
    trans2 = translation_matrix_3d(segment.centroid_3d.xyz);
    rot2 = inv(transformation_matrix_from_vector(segment.centroid_normal, 1));
    
    % scale change
    scale = segment.scale / matches(ii).scale;
    scale_M = scale_matrix_3d(scale);
    
    % creating and applying final transformation
    transf = double(trans2 * rot2 * camera_rot * scale_M * rot1 * trans1);
    translated_match = apply_transformation_3d(matches(ii).xyz, transf);
   
    %subplot(4, 5, ii)
    hold on
    plot3d(translated_match, 'r')
    
end
hold off

%% now must align the match into the original image
imagesc(cloud.depth)
hold on
plot(segment.centroid(1), segment.centroid(2), 'r+')
hold off

%% 2D alignment visualisation
clf
T = cloud.depth;
for ii = 1:10
    
    % computing translations, rotations and scales
    trans1 = translation_matrix(-matches(ii).centroid(1), -matches(ii).centroid(2));
    rot = rotation_matrix(matches(ii).angle);
    trans2 = translation_matrix(segment.centroid(1), segment.centroid(2));
    scale = (matches(ii).median_depth / segment.median_depth ) * ...
        (2) * (segment.scale / matches(ii).scale);
    scale_M = double(scale_matrix(scale));
    
    transf = maketform('affine', (trans2 * rot * scale_M * trans1)');
    [H, W] = size(cloud.depth);
    
    depth = matches(ii).depth;
    depth(isnan(depth)) = 0;
    translated_match = imtransform(depth, transf, 'nearest', 'XData', [1 W], 'YData', [1 H], 'size', size(T));
    T = translated_match/10 + T;
    
end

imagesc(T)




