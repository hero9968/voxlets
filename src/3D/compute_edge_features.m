% Computing all the feature vectors. The aim is to end up with them as
% a single .mat file for each of the 1600 or so objects...

cd ~/projects/shape_sharing/src/3D/
clear
addpath(genpath('.'))
run define_params_3d.m

number_renders = 42;
params.shape_dist.rescaling = 0;
params.shape_dist.num_samples = 20000;

%%
for ii = 101:length(params.model_filelist)

    tic
    
    % output place
    model = params.model_filelist{ii};
    outfile = sprintf(paths.basis_models.fv_file, model);

    % see if edge_fv exists... 
    vars = whos('-file',outfile);
    if ismember('edge_fv', {vars.name})
        disp(['Skipping ' num2str(ii)])
        %continue
    end
    
    % setting up the variables to be filled
    edge_fv = nan(number_renders, 51);

    % loading the data
    modelfile = sprintf(paths.basis_models.combined_file, model);
    load(modelfile, 'renders')

    % loop over each image and combine all the results together
    for jj = 1:number_renders

        depth = renders(jj).depth;

        % project depth to 3D
        tic
        [this_xyz, mask] = reproject_depth(depth, params.half_intrinsics, nan);
            toc
        if size(this_xyz, 1) > 10
            [temp_edge] = edge_angle_fv(~isnan(depth), 50);
            edge_fv(jj, :) = temp_edge';
        end
    end

    %save(outfile, 'shape_dist', 'scale', 'transform_to_origin', 'edge_shape_dist');
    save(outfile, 'edge_fv', '-append')

    disp(['Done ' num2str(ii) ' in ' num2str(toc) 's']);
    
end

