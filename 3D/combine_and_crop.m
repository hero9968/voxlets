% combine depth images (and probably normals) into one file
% I'll crop the images, and include a 'top-left' field.
% Will be interesting to see difference in disk space...

cd ~/projects/shape_sharing/3D/
clear
addpath(genpath('.'))
run define_params_3d.m

number_renders = 42;

%%
for ii = 1:length(params.model_filelist)
    
    % output place
    model = params.model_filelist{ii};
    outfile = sprintf(paths.basis_models.combined_file, model);
    
    if exist(outfile, 'file')
        disp(['Skipping ' num2str(ii)])
        continue
    end
    
    % setting up the variables to be filled
    clear renders

    % loop over each image and combine all the results together
    tic
    for jj = 1:number_renders

        this_depth_name = sprintf(paths.basis_models.rendered, model, jj);
        this_norms_name = sprintf(paths.basis_models.normals, model, jj);
        
        this_depth = load_for_parfor(this_depth_name, 'depth');
        max_depth = max(this_depth(:));
        mask = abs(this_depth-max_depth)< 0.001;
        this_depth(mask) = nan;
                       
        normals = load_for_parfor(this_norms_name, 'normals');
        
        % hack here for normals which were not stripped out correctly...
        if length(normals) == 240*320
            normals(mask, :) = [];
        end
        
        renders(jj).depth = single(this_depth);
        renders(jj).normals = single(normals);
        
    end
    
    save(outfile, 'renders')

    disp(['Done ' num2str(ii) ' in ' num2str(toc) ' s'])
end
