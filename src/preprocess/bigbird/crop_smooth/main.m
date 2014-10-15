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
        
        break
        
        disp([num2str(view_idx)])
      
    end
    disp(['Done model ', num2str(view_idx)])
    
end

%plotNormals(bb_cropped.xyz, bb_cropped.normals, 0.01)