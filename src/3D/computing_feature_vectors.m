% Computing all the feature vectors. The aim is to end up with them as
% a single .mat file for each of the 1600 or so objects...

cd ~/projects/shape_sharing/3D/
clear
addpath(genpath('.'))
run define_params_3d.m

number_renders = 42;
params.shape_dist.rescaling = 0;
params.shape_dist.num_samples = 20000;

%%
for ii = 1:length(params.model_filelist)

    tic
    
    % output place
    model = params.model_filelist{ii};
    outfile = sprintf(paths.basis_models.fv_file, model);

    if exist(outfile, 'file')
        disp(['Skipping ' num2str(ii)])
        continue
    end
    
    % setting up the variables to be filled
    edge_shape_dist = nan(number_renders, size(params.shape_dist.dict, 1));
    shape_dist = nan(number_renders, size(params.shape_dist.dict, 1));
    scale = nan(1, number_renders);
    transform_to_origin = cell(1, number_renders);

    % loading the data
    modelfile = sprintf(paths.basis_models.combined_file, model);
    load(modelfile, 'renders')
    %try 

        % loop over each image and combine all the results together
        for jj = 1:number_renders

            depth = renders(jj).depth;
            norms = renders(jj).normals;
            
            % hack to deal with bad data
            %max_depth = max(depth(:));
            %if abs(max_depth-2)<0.01
            %    normals_to_use = depth~=max_depth;
            %    norms = norms(normals_to_use(:), :);
            %end

            % project depth to 3D
            [this_xyz, mask] = reproject_depth(depth, params.half_intrinsics, nan);

            if size(this_xyz, 1) > 10
                scale(jj) = 1 / estimate_size(this_xyz);
                this_xyz = this_xyz * scale(jj);

                shape_dist(jj, :) = shape_distribution_norms_3d(this_xyz, norms, params.shape_dist);
                %edge_shape_dist(jj, :) = edge_shape_dists(mask, params.shape_dist.edge_dict);
                edge_shape_dist(jj, :) = edge_shape_dists_norms(mask, params.shape_dist.edge_dict);
                
                % transform to origin
                [~, ~, temp] = transformation_to_origin_3d(this_xyz);
                transform_to_origin{jj} = inv(temp);
            end
        end

        save(outfile, 'shape_dist', 'scale', 'transform_to_origin', 'edge_shape_dist');
    %catch
    %    warning(['Could not do number ' num2str(ii)])
    %end
            

    disp(['Done ' num2str(ii) ' in ' num2str(toc) 's']);
    
end



