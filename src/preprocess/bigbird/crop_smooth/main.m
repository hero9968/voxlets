% This is a new script to load in RGB, depth and mask from bigbird
% Then reproject RGB into depth
% Then smooth
% Then crop
% Then save
addpath('~/projects/shape_sharing/src/matlab/matlab_features')
OVERWRITE = false;
matlabpool(6)
%%
base_path = get_base_path();
[views, modelnames] = get_views_models();

for model_idx = 1:length(modelnames)
    
    modelname = modelnames{model_idx};
    
    % creating folder if need be
    model_folder = [base_path, 'bigbird_cropped/', modelname, '/'];
    if ~exist(model_folder, 'dir')
        disp(['Creating ' model_folder])
        mkdir(model_folder)
    end
    
    parfor view_idx = 1:length(views)
        
        view = views{view_idx};
        save_name = [model_folder, view, '.mat'];
        
        % seeing if file already exists
        if ~OVERWRITE && exist(save_name, 'file')
            disp(['Skipping ' save_name])
            continue
        end
        
        % loading, cropping
        try
            bb = load_bigbird(modelname, view);
            bb_cropped = reproject_crop_and_smooth(bb);
            %plot_bb(bb_cropped)

            % saving to disk
            save_file(save_name, bb_cropped);
            
        catch err
            disp(err.message)
        end
	fprintf('%d ', view_idx);
        %disp(num2str(view_idx))
      
    end
    
    disp(['Done model ', modelname, ' ', num2str(model_idx)])
    
end

%plotNormals(bb_cropped.xyz, bb_cropped.normals, 0.01)
