% Computing all the feature vectors. The aim is to end up with them as
% a single .mat file for each of the 1600 or so objects...

cd ~/projects/shape_sharing/3D/
clear
addpath(genpath('.'))
run define_params_3d.m

render_path = '/Users/Michael/projects/shape_sharing/data/3D/basis_models/renders/';
fv_path = '/Users/Michael/projects/shape_sharing/data/3D/basis_models/fv/';
number_renders = 42;

%%
for ii = 1:length(params.model_filelist)

    tic
    
    % output place
    outfile = fullfile(fv_path, [params.model_filelist{ii}, '.mat']);
    if exist(outfile, 'file')
        disp(['Skipping ' num2str(ii)])
        continue
    end
    
    % getting the path to the renders
    model = params.model_filelist{ii};
    render_dir = fullfile(render_path, model);
    depth_names = fullfile(render_dir, 'depth_%d.mat');
    
    shape_dist = nan(number_renders, size(params.shape_dist.dict, 1));
    scale = nan(1, number_renders);

    try 

        % loop over each image and combine all the results together
        for jj = 1:number_renders

            this_name = sprintf(depth_names,jj);
            load(this_name, 'depth');

            % project depth to 3D
            this_xyz = reproject_depth(depth, params.half_intrinsics, 3);

            if size(this_xyz, 1) > 10
                scale(jj) = normalise_scale(this_xyz);
                this_xyz = this_xyz / scale(jj);

                % compute the shape dist
                params.shape_dist.rescaling = 0;
                params.shape_dist.num_samples = 20000;
                shape_dist(jj, :) = shape_distribution_3d(this_xyz, params.shape_dist);
            end
        end

        save(outfile, 'shape_dist', 'scale');
    end

    disp(['Done ' num2str(ii) ' in ' num2str(toc) 's']);
    
end



