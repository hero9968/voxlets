% Compute normals for each training view
% (Not bothering for now...)
cd ~/projects/shape_sharing/3D/
clear
run define_params_3d.m
addpath(genpath('.'))

%% 

for ii = 1:length(params.model_filelist)

    model = params.model_filelist{ii};
    outdir = ['/Users/Michael/projects/shape_sharing/data/3D/basis_models/normals/' model];
    depthdir = ['/Users/Michael/projects/shape_sharing/data/3D/basis_models/renders/' model];
    
    if ~exist(outdir, 'dir')
        disp(['Making directory']);
        mkdir(outdir)
    end
    

    disp(['Doing number ' num2str(ii)]);
    
    tic
    
    for jj = 1:42
        outfile = sprintf([outdir, '/norms_%d.mat'], jj);
        depthfile = sprintf([depthdir, '/depth_%d.mat'], jj);
       
        % deciding if to continue
        if exist(outfile, 'file')
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

