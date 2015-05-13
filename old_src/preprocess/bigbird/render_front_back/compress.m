% resaves all the files, with the better matlab compression
addpath '../crop_smooth/'

base_path = get_base_path();
[views, modelnames] = get_views_models();

%%
for model_idx = 1:length(modelnames)
    
    modelname = modelnames{model_idx};
    
    for view_idx = 1:length(views)
        
        view = views{view_idx};
        this_path = [base_path, 'bigbird_renders/', modelname, '/', view, '_renders.mat'];
        
        if exist(this_path, 'file')
            TT = load(this_path);
            save(this_path, '-struct', 'TT')
        end
    end
end