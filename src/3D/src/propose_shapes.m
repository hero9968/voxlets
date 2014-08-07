% script to propose 3D basis shapes for a specific region
clear
cd ~/projects/shape_sharing/src/3D/src/
addpath(genpath('.'))
addpath(genpath('../../2D/src'))
addpath(genpath('../../common/'))
run ../define_params_3d
load(paths.structured_model_file, 'model')

%% loading the saved and segmented cloud from disk
load(paths.test_dataset.artificial_scene, 'cloud')

%% plotting segments
plot_segment_soup_3d(cloud.rgb, cloud.segment.idxs, cloud.segment.probabilities);

%% Finding all the possible transformations into the scene
params.proposals.proposals_per_region = 2;
params.proposals.feature_vector = 'shape_dist';
params.proposals.load_voxels = false;

all_matches = [];

for seg_idx = 1:size(cloud.segment.idxs, 2);
    
    % for this region, propose matching shapes (+transforms) from the database
    [segment_matches, these_matches] = ...
        propose_matches(cloud.segments{seg_idx}, model, params, paths);
    
    % combining all the matches into one big array
    all_matches = [all_matches, these_matches];
    
    done(seg_idx, size(cloud.segment.idxs, 2));    
end

%% write the matches and the transformations to a yaml file for openvdb to read
yaml_matches = convert_matches_to_yaml(all_matches);
WriteYaml('test.yaml', yaml_matches);

%% 3D visualisation of the regions aligned in
subplot(221)
plot3d(apply_transformation_3d(cloud.xyz, cloud.segment.rotate_to_plane), 'y');
hold on
for ii = 1:length(all_matches)
    plot3d(apply_transformation_3d(all_matches(ii).xyz, all_matches(ii).transformation));
end
hold off
view(122, 90)

%% 3D alignment visualisation of the voxels
% (This is basically equivalnt to what the c++ yaml visualiser should do)
%subplot(222)
clf
plot3d(apply_transformation_3d(cloud.xyz, cloud.segment.rotate_to_plane), 'y');
hold on
xyz = plot_matches_3d(yaml_matches);
hold off
view(122, 90)

%% 3D alignment visualisation of the openvdb voxels
% (run ./yaml > final_locations.txt in the correct folder first)
subplot(223)
plot3d(apply_transformation_3d(cloud.xyz, cloud.segment.rotate_to_plane), 'y');
hold on
openvdb = load('/Users/Michael/projects/shape_sharing/src/tools/openvdb_tests/final_locations.txt');
plot3d(openvdb(:, 1:3)/100)
hold off
view(122, 90)

%% trying to do nice plotting of voxel slices in 2D space
for ii = 0.05:0.05:0.4
    plot_voxel_scene(cloud, openvdb, ii)
   
    drawnow
    pause(0.1)
    
   
end

%% alternative visualisation where the voxels are converted first...



%% Checking alignment
%plot_matches_3d(matches);

%% sorting out the transformation fuckups
openvdb = load('/Users/Michael/projects/shape_sharing/src/tools/openvdb_tests/final_locations.txt');
%openvdb = openvdb(:, [2, 1, 3]);
matlab = [xyz{1}{1}(:, :) * 100];% xyz{1}{2}(:, :) * 100];
whos matlab openvdb
voxeldiff(openvdb(:, 1:3), matlab)
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



%% checking the voxel align - should be able to run this by itself
model_name = params.model_filelist{100};
matlab_vox = load_vox(model_name);

voxel_filename = sprintf('/Users/Michael/projects/shape_sharing/src/3D/src/voxelisation/vdb_convert/%s.txt', model_name);
openvdb_vox = load(voxel_filename);

plot3d(matlab_vox, 'b');
hold on
plot3d(openvdb_vox, 'r');
hold off

