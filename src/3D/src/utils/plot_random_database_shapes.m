%% plotting some random database shapes
addpath(genpath('../../common/'))
addpath ../../2D/src/utils

for ii = 1:25
    
    this_idx = randi(length(model.all_model_idx));
    this.model_idx = model.all_model_idx(this_idx);
    this.model = params.model_filelist{this.model_idx};
    this.view = model.all_view_idx(this_idx);
    this.path = sprintf(paths.basis_models.rendered, this.model, this.view);
    load(this.path, 'depth')
    max_depth = max(depth(:));
    depth(abs(depth-max_depth)<0.01) = 0;
    depth = boxcrop_2d(depth);
    depth(depth==0) =nan;
    subaxis(5, 5, ii, 'Margin',0, 'Spacing', 0)
    imagesc(depth)
    axis image off
    colormap(flipud(gray))    
end
set(gcf, 'color', [1, 1, 1])