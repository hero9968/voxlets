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
[cloud.segment.idxs, ~, cloud.segment.probabilities, ~, cloud.segment.rotate_to_plane] = ...
    segment_soup_3d(cloud, params.segment_soup);

%% plotting segments
plot_segment_soup_3d(cloud.rgb.^0.2, cloud.segment.idxs, cloud.segment.probabilities);

%% plot 3d after rotation...
temp_xyz = apply_transformation_3d(cloud.xyz, cloud.segment.rotate_to_plane);
plot3d(temp_xyz)
view(0, 0)

%% Finding all the possible transformations into the scene
proposals_per_region = 1;
all_matches = [];
for seg_index = 1:size(cloud.segment.idxs, 2);
    
    segment = extract_segment(cloud, cloud.segment.idxs(:, seg_index), params);
    segment_matches = propose_matches(segment, model, proposals_per_region, 'shape_dist', params, paths);
    
    for ii = 1:proposals_per_region

        % creating and applying final transformation
        transf = double(cloud.segment.rotate_to_plane * segment.transforms.final_M * segment_matches(ii).transforms.final_M);
        vox_transf = double(transf * segment_matches(ii).transforms.vox_inv * params.voxelisation.T_vox);
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
    
    done(seg_index, size(cloud.segment.idxs, 2));    
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
        transform.R = transM(1:3, 1:3);% * [0, 1, 0; -1, 0, 0; 0, 0, 1];
        transform.weight = 1;
        transform.region = all_matches(this_matches(jj)).region;
        match.transform{end+1} = transform;
    end
    matches{end+1} = match;
end

%matches{1}.transform([1:7, 9:end]) = []
% writing to YAML file
%matches{1}.transform{1}.R = eye(3) / 100;
%matches{1}.transform{1}.T = 0 * matches{1}.transform{1}.T;
WriteYaml('test.yaml', matches, 0);
% plotting the closest matches
%plot_matches(matches, 20, segment.mask, params, paths)


%% 3D visualisation of the regions
subplot(121)
plot3d(apply_transformation_3d(cloud.xyz, cloud.segment.rotate_to_plane), 'y');
hold on
for ii = 1:length(all_matches)
    plot3d(apply_transformation_3d(all_matches(ii).xyz, all_matches(ii).transformation));
end
hold off
view(122, 90)

%% 3D alignment visualisation of the voxels
% (This is basically equivalnt to what the c++ yaml visualiser should do)
subplot(122)
plot3d(apply_transformation_3d(cloud.xyz, cloud.segment.rotate_to_plane), 'y');
hold on
xyz = plot_matches_3d(matches);
hold off
view(122, 90)



%% Checking alignment
%plot_matches_3d(matches);

%% sorting out the transformation fuckups
openvdb = load('/Users/Michael/projects/shape_sharing/src/tools/openvdb_tests/final_locations.txt');
%openvdb = openvdb(:, [2, 1, 3]);
matlab = [xyz{1}{1}(:, :) * 100];% xyz{1}{2}(:, :) * 100];
whos matlab openvdb
voxeldiff(openvdb, matlab)
view(-45, 90)

%%
plot3d(round(matlab), 'r');
hold on
plot3d(openvdb+0.5, 'g');
hold off

%% basic fuckery
openvdb = load('/Users/Michael/projects/shape_sharing/src/3D/src/voxelisation/vdb_convert/base_file.txt');
%openvdb = openvdb(:, [2, 1, 3]);
matlab = load_vox('1046b3d5a381a8902e5ae31c6c631a39');
%whos matlab openvdb
voxeldiff(openvdb, matlab)
view(-45, 30)


