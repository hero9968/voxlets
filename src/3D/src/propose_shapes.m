% script to propose 3D basis shapes for a specific region
clear
cd ~/projects/shape_sharing/src/3D/src/
addpath(genpath('.'))
addpath(genpath('../../2D/src'))
addpath(genpath('../../common/'))
run ../define_params_3d
load(paths.structured_model_file, 'model')

%% loading in some of the ECCV dataset, normals + segmentation
cloud = loadpgm_as_cloud('~/projects/shape_sharing/data/3D/scenes/first_few_render_noisy00000.pgm', params.full_intrinsics);
[cloud.normals, cloud.curvature] = normals_wrapper(cloud.xyz, 'knn', 50);
[idxs, idxs_without_nans, probabilities, all_idx] = segment_soup_3d(cloud, params.segment_soup);

%% plotting segments
plot_segment_soup_3d(cloud.rgb.^0.2, idxs, probabilities);

%% Finding all the possible transformations into the scene
all_matches = [];
for seg_index = [1, 10]%:size(idxs, 2);
    
    segment = extract_segment(cloud, idxs(:, seg_index), params);
    segment_matches = propose_matches(segment, model, 20, 'shape_dist', params, paths);
    
    for ii = 1:20

        % creating and applying final transformation
        transf = double(segment.transforms.final_M * segment_matches(ii).transforms.final_M);
        vox_transf = double(segment.transforms.final_M * segment_matches(ii).transforms.final_M * segment_matches(ii).transforms.vox_inv * params.voxelisation.T_vox);
        translated_match = apply_transformation_3d(segment_matches(ii).xyz, transf);

        % combining some of the most vital info into a new structure
        this_match.object_name = segment_matches(ii).model.name;
        this_match.transformation = transf;
        this_match.vox_transformation = vox_transf;
        this_match.region = seg_index;
        this_match.xyz = segment_matches(ii).xyz;
        this_match.vox_xyz = segment_matches(ii).vox_xyz;
        segment_matches(ii).xyz(:, 2:3) = segment_matches(ii).xyz(:, 2:3);
        all_matches = [all_matches, this_match];
    end
    
    done(seg_index, size(idxs, 2));    
end

%% Condense all the transformations into one nice structure heirarchy
unique_objects = unique({all_matches.object_name});
matches = {};

for ii = 1:length(unique_objects)    
    this_matches = find(ismember({all_matches.object_name}, unique_objects{ii}));
    match.name = unique_objects{ii};
    match.transform = {};

    for jj = 1:length(this_matches)
        transM = all_matches(this_matches(jj)).vox_transformation;
        transform.T = transM(1:3, 4)';
        transform.R = transM(1:3, 1:3);
        transform.weight = 1;
        transform.region = all_matches(this_matches(jj)).region;
        match.transform{end+1} = transform;
    end
    matches{end+1} = match;
end

%matches{1}.transform([1:7, 9:end]) = []
% writing to YAML file
WriteYaml('test.yaml', matches, 0);
% plotting the closest matches
%plot_matches(matches, 20, segment.mask, params, paths)

%% 3D alignment visualisation of all matches
% (This is basically equivalnt to what the c++ yaml visualiser should do)
plot3d(cloud.xyz, 'y');
hold on
plot_matches_3d(matches);
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
    trans1 = translation_matrix(-segment_matches(ii).centroid(1), -segment_matches(ii).centroid(2));
    rot = rotation_matrix(segment_matches(ii).angle);
    trans2 = translation_matrix(segment.centroid(1), segment.centroid(2));
    scale = (segment_matches(ii).median_depth / segment.median_depth ) * ...
        (2) * (segment.scale / segment_matches(ii).scale);
    scale_M = double(scale_matrix(scale));
    
    transf = maketform('affine', (trans2 * rot * scale_M * trans1)');
    [H, W] = size(cloud.depth);
    
    depth = segment_matches(ii).depth;
    depth(isnan(depth)) = 0;
    translated_match = imtransform(depth, transf, 'nearest', 'XData', [1 W], 'YData', [1 H], 'size', size(T));
    T = translated_match/10 + T;
    
end

imagesc(T)
