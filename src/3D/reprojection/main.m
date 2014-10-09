% This is a new script to load in RGB, depth and mask from bigbird
% Then reproject RGB into depth
% Then smooth
% Then crop
% Then save

modelnames = {'cinnamon_toast_crunch'};
views = {'NP1_24'};

for model_idx = 1:length(modelnames)
    modelname = modelnames{model_idx};
    for view_idx = 1:length(views)
        view = views{view_idx};
        %%
        bb = load_bigbird(modelname, view);
        %%
        bb = reproject_crop_and_smooth(bb);
        
        %bigbird = crop_bb(bigbird);
        
        % now save
        
        
        
    end
    
end