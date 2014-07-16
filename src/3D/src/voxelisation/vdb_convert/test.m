%% loading the original voxels
T = load(voxel_path);
original_voxels = T.vol>40;

%% running the conversion
%torun = './txt2vdb /Users/Michael/projects/shape_sharing/data/3D/basis_models/voxelised_text/1046b3d5a381a8902e5ae31c6c631a39.txt > temp.txt'
%system(torun);

%%
converted_indices = load('temp.txt') + 1; % convert to 1-indexing
converted_voxels = zeros(100, 100, 100);

%%
for ii = 1:length(converted_indices)
    converted_voxels(converted_indices(ii, 1), converted_indices(ii, 2), converted_indices(ii, 3)) = 1;
end

%%
for ii = 1:100
    subplot(131)
    imagesc(squeeze(original_voxels(ii, :, :)))
    axis image
    subplot(132)
    imagesc(squeeze(converted_voxels(ii, :, :)))
    axis image
    subplot(133)
    imagesc(squeeze(converted_voxels(ii, :, :) - original_voxels(ii, :, :)))
    axis image
    drawnow
    pause(0.1)
end
%%
sum(abs(converted_voxels(:) - original_voxels(:)))