
        
% do edges and spider separately... these are more likely to change!
% This is a new script to load in RGB, depth and mask from bigbird
% Then reproject RGB into depth
% Then smooth
% Then crop
% Then save
cd ~/projects/shape_sharing/src/edges_and_compass/
addpath('~/projects/shape_sharing/src/preprocess/bigbird/crop_smooth/')
addpath('~/projects/shape_sharing/src/matlab/matlab_features/')
OVERWRITE = false;
matlabpool(3)
%%
base_path = get_base_path();
[views, modelnames] = get_views_models();
%modelnames = {'nutrigrain_apple_cinnamon'}
%views = {'NP2_96'}
%% 24

for model_idx = 1:length(modelnames)
    
    modelname = modelnames{model_idx};
    
    model_folder = [base_path, 'bigbird_cropped/', modelname, '/'];
    
    parfor view_idx = 1:length(views)
        
        view = views{view_idx};
        load_name = [model_folder, view, '.mat'];
        save_name = [model_folder, view, '_spider.mat'];
        
        % seeing if file already exists
        if ~OVERWRITE && exist(save_name, 'file')
            disp(['Skipping ' save_name])
            continue
        end
        
        % loading, cropping
        try
            bb = load(load_name);
            
            % edges from the mask
            edges = edge(bb.mask);
            se = strel('disk', 4);
            edges = imdilate(edges, se);
            
            % normals from the noisy data
            %keyboard
            norms = normals_wrapper(bb.xyz, 'knn', 100);
            
            % identifying flying pixels
            %xyz_length = sqrt(sum(bb.xyz.^2, 2));
            %normalised_xyz = bb.xyz ./ repmat(xyz_length, 1, 3);
            %angle_xyz = dot(normalised_xyz, norms, 2);
            %cos_angle = abs(reshape(angle_xyz, size(bb.grey)));
            
            
            % spider featres
            spider = spider_wrapper(bb.xyz, norms, edges, bb.mask, bb.T_K_rgb(1));
 
            % saving to disk
            save_spider(save_name, spider, norms, edges);
            
        catch err
            disp(err.message)
        end
        disp(num2str(view_idx))
      
    end
    
    disp(['Done model ', modelname, ' ', num2str(model_idx)])
    
end

%plotNormals(bb_cropped.xyz, bb_cropped.normals, 0.01)

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
        xyz_normalised = sqrt(sum(bb_cropped.xyz.^2, 2));
        norm_xyz = bb_cropped.xyz ./ repmat(xyz_normalised, 1, 3);
        angle_xyz = dot(norm_xyz, bb_cropped.normals, 2);
        cos_angle = 1-abs(reshape(angle_xyz, size(bb_cropped.clean.edges)));
        imagesc(cos_angle>0.3);
        axis image
        colorbar
                
        %% now save to 
        