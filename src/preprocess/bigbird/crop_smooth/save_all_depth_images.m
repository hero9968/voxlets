clear

%% writing out depths

base_path = get_base_path();
[views, modelnames] = get_views_models();
modelnames = modelnames(end-21:end);

for model_idx = 1:length(modelnames)
    
    modelname = modelnames{model_idx};
    obj_path = [base_path, 'bigbird/', modelname];
 
    for view_idx = [1:5:46]

        view = views{view_idx};
        
        depth = h5read([obj_path, '/' , view, '.h5'], '/depth')';
        depth = single(depth) / 10000;
        
        % writing image to a file
        outfilename = [base_path, 'bigbird_depths/', modelname, '_', view, '_depth.png'];
        imwrite(uint16(depth*1000), outfilename)
        
    end
    modelname
end