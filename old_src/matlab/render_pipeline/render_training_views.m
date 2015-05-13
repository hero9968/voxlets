% a script to render the models from different angles, using the python
% script.
%
clear
cd ~/projects/shape_sharing/src/3D/
run define_params_3d.m

%% generating a reduced filelist which only consists of files which are small enough
% (This has now been incorporated into the main model list)
%{
fid = fopen('temp.txt', 'w');

for ii = 1:length(params.model_filelist)
    P = dir(fullfile(paths.basis_models.centred, [params.model_filelist{ii} '.obj']));
    
    if P.bytes < 1e6
        fprintf(fid, '%s\n', params.model_filelist{ii});
    end
end

fclose(fid);
%}

%%
for ii = params.files_to_use
    
    model = params.model_filelist{ii};
    
    % loading meta file
    meta_file = [paths.basis_models.centred '/' params.model_filelist{ii} '.mat'];
    load(meta_file, 'aabb');
    %distance = aabb.diag / ( 2 * tand(43/2) ) % (vertical FOV is 43 degrees)
    distance = 1; % as all shapes should be nice and rescaled
    
    % creating output directory if it doesn't exist
    outdir = ['/Users/Michael/projects/shape_sharing/data/3D/basis_models/renders/' model];
    
    if ~exist(outdir, 'dir')
        disp(['Making directory']);
        mkdir(outdir)
    end
    
	% deciding if to continue
    if length(dir(outdir)) >= 44
        disp(['Skipping number ' num2str(ii)]);
        %continue;
    end
    
    disp(['Doing number ' num2str(ii)]);
    
    % calling python script to render the sequence
    system_call = ['python src/model_render/getDepthSequence.py ' model ' ' num2str(distance) ' 1 42'];
    %system_call
    [A, B] = system(system_call);
    pause(1)
    
    disp(['Done ' num2str(ii)]);
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

