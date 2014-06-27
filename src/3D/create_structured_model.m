% script to load all the feature vectors and save a model

clear
cd ~/projects/shape_sharing/src/3D
addpath(genpath('src'))
addpath(genpath('../common'))
define_params_3d

%%
%shape_dists = cell(length(params.model_filelist), 1);
training = [];

for ii = 1:length(params.model_filelist)
    this_path = sprintf(paths.basis_models.fv_file, params.model_filelist{ii});
    load(this_path, 'shape_dist', 'edge_shape_dist', 'edge_fv');
    training(ii).shape_dist = shape_dist;
    training(ii).edge_shape_dist = edge_shape_dist;
    training(ii).edge_angles_fv = edge_fv;
    N = size(shape_dist, 1);
    training(ii).model_name = params.model_filelist{ii};
    training(ii).model_idx = ii * ones(1, N);
    training(ii).view_idx = 1:N;
    done(ii, length(params.model_filelist))
end
%%
model.all_shape_dists = cell2mat({training.shape_dist}');
model.all_edge_shape_dists = cell2mat({training.edge_shape_dist}');
model.all_model_idx = cell2mat({training.model_idx})';
model.all_view_idx = cell2mat({training.view_idx})';
model.all_edge_angles_fv = cell2mat({training.edge_angles_fv}');
save(paths.structured_model_file, 'model')
disp('Saved model')