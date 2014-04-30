% a script to render the models from different angles, using the python
% script.
%

for ii = 1:length(ply_consts.model_names)
    
    model = params.model_filelist{ii};
    
    % loading meta file
    meta_file = [paths.basis_models.centred '/' params.model_filelist{ii} '.mat'];
    load(meta_file, 'aabb');
    distance = aabb.diag / ( 2 * tand(43/2) ) % (vertical FOV is 43 degrees)
    
    % creating output directory if it doesn't exist
    outdir = ['/Users/Michael/projects/shape_sharing/data/3D/basis_models/renders/' model];
    if ~exist(outdir, 'dir')
        mkdir outdir
    end
    
    % calling python script to render the sequence
    system_call = ['python getDepthSequence.py ' model ' ' num2str(1.1*distance) ' 1 42'];
    %[A, B] = system(system_call);
    
    ii
end


%% display the depth maps for a certain object

model_idx = 1;
model_name = params.model_filelist{model_idx};
    
outdir = ['/Users/Michael/projects/shape_sharing/data/3D/basis_models/renders/' model_name];

for ii = 1:42

    % loading the depth image
    load([outdir, '/depth_' num2str(ii) '.mat'], 'depth');
    
    % plotting the depth image in a subplot
    subplot(6, 7, ii);
    imagesc(depth);
    axis image off
    title(num2str(ii));
    
end

