
        
% do edges and spider separately... these are more likely to change!


% spider feature?

        
                
        %% creating clean edges (won't need this when it comes to it)
        t_mask = isnan(bb_cropped.front_render);
        se = strel('disk', 3);
        t_mask = imdilate(imopen(t_mask, se), se);
        bb_cropped.clean.edges = edge(t_mask);
        
        %% plotting
        subplot(131)
        imagesc(isnan(bb_cropped.front_render))
        axis image
        subplot(132)
        imagesc(t_mask)
        axis image
        subplot(133)
        imshow(bb_cropped.clean.edges)
        
        %% running the spider feature mex file
        tic
        se = strel('disk', 3);
        dilated_edges = imdilate(bb_cropped.clean.edges, se);
        sp = spider_wrapper(bb_cropped.clean.xyz, bb_cropped.clean.normals, dilated_edges, bb_cropped.T.K_rgb(1));
        toc
        
        %% displayin the spider
        for ii = 1:12
            subplot(4, 3, ii)
            imagesc(sp(:, :, ii))
            axis image
            colormap(jet)
            colorbar
        end
       
        %% identifying flying pixels
        xyz_norm = sqrt(sum(bb_cropped.xyz.^2, 2));
        norm_xyz = bb_cropped.xyz ./ repmat(xyz_norm, 1, 3);
        angle_xyz = dot(norm_xyz, bb_cropped.normals, 2);
        cos_angle = 1-abs(reshape(angle_xyz, size(bb_cropped.clean.edges)));
        imagesc(cos_angle>0.3);
        axis image
        colorbar
                
        %% now save to disk
        
        