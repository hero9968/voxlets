% This is a new script to load in RGB, depth and mask from bigbird
% Then reproject RGB into depth
% Then smooth
% Then crop
% Then save

addpath('/Users/Michael/builds/research_code/structured_edge_detection/release')
addpath('/Users/Michael/projects/shape_sharing/src/matlab/matlab_features')
addpath(genpath('/Users/Michael/projects/shape_sharing/src/common/toolbox'))
%load('modelNyuRgbd.mat')

%%
modelnames = {'aunt_jemima_original_syrup'};
%modelnames = {'cheez_it_white_cheddar'};
%views = {'NP5_336'};
views = {'NP2_336'};%'NP3_336', 'NP1_336', , 'NP3_336', 'NP4_336', 'NP5_336'};

for model_idx = 1:length(modelnames)
    modelname = modelnames{model_idx};
    
    for view_idx = 1:length(views)
        
        view = views{view_idx};

        % loading, cropping
        bb = load_bigbird(modelname, view);
        bb_cropped = reproject_crop_and_smooth(bb);
        %plot_bb(bb_cropped)
        
        % computing xyz and normals
        bb_cropped.xyz = reproject_3d(bb_cropped.depth, bb_cropped.T.K_rgb, bb_cropped.aabb([1, 3]));
        bb_cropped.normals = normals_wrapper(bb_cropped.xyz, 'knn', 100);
        
        %% computing clean xyz and normals
        bb_cropped.clean.xyz = reproject_3d(bb_cropped.front_render, bb_cropped.T.K_rgb, bb_cropped.aabb([1, 3]));
        bb_cropped.clean.normals = normals_wrapper(bb_cropped.clean.xyz, 'knn', 100);
                
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
        
        
        disp([num2str(view_idx)])
      
    end
    disp(['Done model ', num2str(view_idx)])
    
end

%plotNormals(bb_cropped.xyz, bb_cropped.normals, 0.01)