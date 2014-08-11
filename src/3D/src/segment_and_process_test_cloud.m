% a script to process each test cloud, ultimately saving them to disk
%
clear
cd ~/projects/shape_sharing/src/3D/src/
addpath(genpath('.'))
addpath(genpath('../../2D/src'))
addpath(genpath('../../common/'))
run ../define_params_3d

cloud_pgm_path = '~/projects/shape_sharing/data/3D/scenes/first_few_render_noisy00000.pgm';

%% loading in some of the ECCV dataset, normals + segmentation
%cloud = loadpgm_as_cloud(cloud_pgm_path, params.full_intrinsics);
cloud = structuredcloud(cloud_pgm_path);

%%
[cloud.normals, cloud.curvature] = normals_wrapper(cloud.xyz, 'knn', 50);

[cloud.segmentsoup, ~, cloud.plane_rotate] = ...
    segment_soup_3d(cloud, params.segment_soup);

%% extracting the segments
for seg_idx = 1:size(cloud.segment.idxs, 2);
    
    cloud.segments{seg_idx} = ...
        extract_segment(cloud, cloud.segment.idxs(:, seg_idx), params);
    cloud.segments{seg_idx}.seg_index = seg_idx;
    cloud.segments{seg_idx} = rmfield(cloud.segments{seg_idx}, 'cloud');
    
    done(seg_idx, size(cloud.segment.idxs, 2));
end

%% saving
save(paths.test_dataset.artificial_scene, 'cloud')