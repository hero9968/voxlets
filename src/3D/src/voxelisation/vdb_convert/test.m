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
%% test - should be zero!
sum(abs(converted_voxels(:) - original_voxels(:)))

%% New test...
% Aim is to load a voxelgrid and plot on the same axes as the projected
% views (see e.g. alignment_check.m or whatever it is...)

% Should I do this here or in openvdb?
% probably want to get out the world coordinates of occupied voxels from the test
% program or something
% Alternatively could do all in MATLAB first .... probably best.
addpath ../../plotting/
% now temp.txt contains the transformed voxels
A = load('temp.txt');
plot3d(A)


