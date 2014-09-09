% script to save a jpg thumbnail of each object to a folder, for ease of human
% viewing
clear
cd ~/projects/shape_sharing/src/3D/render_pipeline
run define_params_3d

%%
%view_idx= 32;  42, 21, 26
view_idxs = 1:42;
back_render_path = '/Users/Michael/projects/shape_sharing/data/3D/basis_models/render_backface/%s/depth_%d.mat'
%back_render_path = '/Users/Michael/projects/shape_sharing/data/3D/basis_models/renders/%s/depth_%d.mat'

for ii = 1:1601
    
    for jj = 1:length(view_idxs);
        thispath = sprintf(back_render_path, params.model_filelist{ii}, view_idxs(jj));
        load(thispath, 'depth')
        mask = abs(depth-0.1)<1e-8;
        depth(mask) = nan;
        save(thispath, 'depth')
    end
    
    %f_im = cell2mat(reshape(D, 2, 2));
    %h = imagesc(f_im)
    %colormap(jet(200))
    %set(h, 'AlphaData', ~isnan(f_im))
    
    %imagesc(rgb)
    %axis image
    %drawnow
    
    %pathout = sprintf('/Users/Michael/projects/shape_sharing/data/3D/basis_models/thumbnails/%s.png', params.model_filelist{ii});
    %imwrite(rgb, pathout);
    
    ii
end

