% Computing all the feature vectors. The aim is to end up with them as
% a single .mat file for each of the 1600 or so objects...

cd ~/projects/shape_sharing/3D/
clear
addpath(genpath('.'))
run define_params_3d.m

fv_path = '/Users/Michael/projects/shape_sharing/data/3D/basis_models/fv/';
number_renders = 42;

%%
for ii = 1:length(params.model_filelist)

    tic
    
    % output place
    outfile = sprintf(paths.basis_models.fv_file, params.model_filelist{ii});
    if exist(outfile, 'file')
        disp(['Skipping ' num2str(ii)])
        continue
    end
    
    % setting up the variables to be filled
    shape_dist = nan(number_renders, size(params.shape_dist.dict, 1));
    scale = nan(1, number_renders);
    transform_to_origin = cell(1, number_renders);

    try 

        % loop over each image and combine all the results together
        for jj = 1:number_renders

            this_name = sprintf(paths.basis_models.rendered, model, jj);
            load(this_name, 'depth');

            % project depth to 3D
            this_xyz = reproject_depth(depth, params.half_intrinsics, 3);

            if size(this_xyz, 1) > 10
                scale(jj) = normalise_scale(this_xyz);
                this_xyz = this_xyz * scale(jj);

                % compute the shape dist
                params.shape_dist.rescaling = 0;
                params.shape_dist.num_samples = 20000;
                shape_dist(jj, :) = shape_distribution_3d(this_xyz, params.shape_dist);
                
                [~, ~, temp] = transformation_to_origin_3d(this_xyz);
                transform_to_origin{jj} = inv(temp);
            end
        end

        save(outfile, 'shape_dist', 'scale', 'transform_to_origin');
    end

    disp(['Done ' num2str(ii) ' in ' num2str(toc) 's']);
    
end



