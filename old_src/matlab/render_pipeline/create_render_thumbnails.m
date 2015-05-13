% script to save a jpg thumbnail of each object to a folder, for ease of human
% viewing
clear
cd ~/projects/shape_sharing/src/3D/src/
run ../define_params_3d

%%
%view_idx= 32;  42, 21, 26
clf
view_idxs =[42, 25, 32, 21 ];
for ii = 1:1601
    D = {};
    
    for jj = 1:length(view_idxs);
        path = sprintf(paths.basis_models.rendered, params.model_filelist{ii}, view_idxs(jj));
        load(path, 'depth')
        D{jj} = depth;
    end
    
    f_im = cell2mat(reshape(D, 2, 2));
    %h = imagesc(f_im)
    %colormap(jet(200))
    %set(h, 'AlphaData', ~isnan(f_im))
    
    n = size(nanunique(reshape(f_im,size(f_im,1)*size(f_im,2),size(f_im,3))),1);
    idx = gray2ind(f_im,n);
    idx(isnan(f_im)) = nanmin(idx(~isnan(f_im))) - 50;
    idx = idx - min(idx(:));
    m = length(unique(idx(:)));
    rgb = ind2rgb(idx,[1, 1, 1; jet(m)]);

    %imagesc(rgb)
    %axis image
    %drawnow
    
    pathout = sprintf('/Users/Michael/projects/shape_sharing/data/3D/basis_models/thumbnails/%s.png', params.model_filelist{ii});
    imwrite(rgb, pathout);
    
    ii
end

