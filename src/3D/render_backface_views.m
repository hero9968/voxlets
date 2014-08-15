% a script to render the models from different angles, using the python
% script.
%
clear
addpath ../common
cd ~/projects/shape_sharing/src/3D/
run define_params_3d.m

%%
for ii = 1:params.files_to_use
    
    model = params.model_filelist{ii};

    distance = 1; % as all shapes should be nice and rescaled
    
    % creating output directory if it doesn't exist
    outdir = ['/Users/Michael/projects/shape_sharing/data/3D/basis_models/render_backface/' model];
    
    if ~exist(outdir, 'dir')
        disp(['Making directory']);
        mkdir(outdir)
    end
    
	% deciding if to continue
    if length(dir(outdir)) >= 44
        disp(['Skipping number ' num2str(ii)]);
        continue;
    end
    
    disp(['Doing number ' num2str(ii)]);
    
    % calling python script to render the sequence
    system_call = ['python src/model_render/getDepthSequenceBackFace.py ' model ' ' num2str(distance) ' 1 42'];
    %system_call
    [A, B] = system(system_call);
    %pause(1)
    
    disp(['Done ' num2str(ii)]);
end


