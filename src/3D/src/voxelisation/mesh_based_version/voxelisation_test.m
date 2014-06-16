% script to do voxelisation
cd ~/projects/shape_sharing/3D/model_render/voxelisation/
addpath ../../plotting/
run ../../define_params_3d.m

%%

% deciding which file to use
filename = params.model_filelist{6};
fullpath = fullfile(paths.basis_models.centred, [filename, '.obj']);

% doing voxelisation
voxel_size = 0.01;
tic
vox = voxelise(fullpath, voxel_size);
toc

% stats
size(vox)
sum(vox(:))
sum(vox(:))/numel(vox)

% visualising
clf
vol3d('CData', vox);
axis image