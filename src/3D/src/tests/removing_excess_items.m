% removing excess items
% a test script to look into removing duplicate views from objects
% In some way though we don't really want to remove duplicate views...
% really should be using a structured forest to do the prediction. 

% BUT for now maybe I should just try to do this...
clear
cd ~/projects/shape_sharing/src/3D/src/
addpath(genpath('.'))
addpath(genpath('../../2D/src'))
addpath(genpath('../../common/'))
run ../define_params_3d
load(paths.structured_model_file, 'model')

%% loading a model
close all
model_idx = 121;
fidxs = find(model.all_model_idx == model_idx);

for ii = 1:length(fidxs)
    this.model = params.model_filelist{model_idx};
    this.view = model.all_view_idx(fidxs(ii));
    this.path = sprintf(paths.basis_models.rendered, this.model, this.view);
    load(this.path, 'depth')    
    subplot(7, 7, ii);
    plot_depth(depth);
    title(num2str(ii))
end

figure
%imagesc(sd)

sd = model.all_shape_dists(fidxs, :);
esd = model.all_edge_shape_dists(fidxs, :);
T = linkage([sd, esd]);
dendrogram(T, 42)