function vox = load_vox(model_name)

voxel_filename = sprintf('/Users/Michael/projects/shape_sharing/data/3D/basis_models/voxelised/%s.mat', model_name);
vox_struct = load(voxel_filename);
T = double(vox_struct.sparse_volume(:));
%V = full_3d(double(vox_struct.res) * [1, 1, 1], T);
vox_size = double(vox_struct.res) * [1, 1, 1];
[j, i, k] = ind2sub(vox_size, T);
vox = [i, j, k];



% full find function on the 3d data returning all the indices
function [a, b, c] = fullfind(V)

[a, temp] = find(V);
[b, c] = ind2sub([size(V, 2), size(V, 3)], temp);