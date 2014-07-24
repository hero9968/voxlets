% Compute normals for each training view
% (Not bothering for now...)
cd ~/projects/shape_sharing/src/3D/
clear
run define_params_3d.m
addpath(genpath('.'))

%% 

for ii = params.files_to_use

    model = params.model_filelist{ii};
    outdir = sprintf(paths.basis_models.normals_dir, model);
    
    if ~exist(outdir, 'dir')
        disp(['Making directory']);
        mkdir(outdir)
    end
    
    disp(['Doing number ' num2str(ii)]);
    
    tic
    
    for jj = 1:42
        outfile = sprintf(paths.basis_models.normals, model, jj);
        depthfile = sprintf(paths.basis_models.rendered, model, jj);
       
        % deciding if to continue
        if exist(outfile, 'file') && ~params.overwrite
            disp(['Skipping number ' num2str(ii)]);
            continue;
        end
        
        load(depthfile, 'depth');
        max_depth = depth(1);
        xyz = reproject_depth(depth, params.half_intrinsics, max_depth);
        [normals, curvature] = normals_wrapper(xyz, 'knn', 50);
        
        save(outfile, 'normals', 'curvature')

    end
    
    disp(['Done ' num2str(ii) ' in ' num2str(toc) 's']);
    
end

