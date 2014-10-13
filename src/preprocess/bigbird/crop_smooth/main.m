% This is a new script to load in RGB, depth and mask from bigbird
% Then reproject RGB into depth
% Then smooth
% Then crop
% Then save

addpath('/Users/Michael/builds/research_code/structured_edge_detection/release')
addpath(genpath('/Users/Michael/projects/shape_sharing/src/common/toolbox'))
%load('modelNyuRgbd.mat')


%modelnames = {'aunt_jemima_original_syrup'};
modelnames = {'advil_liqui_gels'};
%views = {'NP5_336'};
views = {'NP3_336', 'NP1_336', 'NP2_336', 'NP3_336', 'NP4_336', 'NP5_336'};



for model_idx = 1:length(modelnames)
    modelname = modelnames{model_idx};
    for view_idx = 1:length(views)
        view = views{view_idx};
        %
        bb = load_bigbird(modelname, view);
        bb_cropped = reproject_crop_and_smooth(bb);
        plot_bb(bb_cropped)
        break
subplot(121)
imagesc(bb.depth)
axis image
subplot(122)

%temp =10*flipud(bb.front_render);
temp(isnan(temp)) = 0;
imagesc(temp + bb.depth)
axis image
pause(1)
        %
        %%
%{
         %bigbird = crop_bb(bigbird);
        subplot(221)
        imagesc(bb_cropped.rgb)
        axis image
        subplot(222)
        imagesc(bb_cropped.depth)
        axis image
        subplot(223)
        imagesc(edge(bb_cropped.depth, 'canny', 0.2))
        axis image
        subplot(224)
        %TT = cat(3, im2double(bb_cropped.rgb), bb_cropped.depth);
        %TT = bb_cropped.depth;
        %t_edges = edgesDetect(TT, model);
        %imagesc(t_edges )
        imagesc(edge(bb_cropped.mask))
%}
        axis image
        % now save
        

        disp([num2str(view_idx)])
      
    end
    disp(['Done model ', num2str(view_idx)])
    
end