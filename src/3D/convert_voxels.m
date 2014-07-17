% script to convert the matlab voxel carving results to cleaner voxel
% outputs, in txt format to be read by a C++ program or other...
clear
close all
cd ~/projects/shape_sharing/src/3D
define_params_3d
threshold = 40;

%%
for ii = 1:length(params.model_filelist)
    
    voxel_path = [paths.basis_models.voxelised, params.model_filelist{ii}, '.mat'];
    text_path = [paths.basis_models.voxelised_text, params.model_filelist{ii}, '.txt'];
    
    if exist(text_path, 'file')
        disp(['Skipping ' num2str(ii)])
        continue;
    end
    
    % loading data and converting to new representation
    file_contents = load(voxel_path);
    temp_vol = single(file_contents.vol);
    filled_locations = find(temp_vol>threshold);
    
    % writing to file
    fid = fopen(text_path, 'w');
    assert(fid~=-1, 'cannot open proper path')
    fprintf(fid, '%d %d %d\n', size(temp_vol, 1), size(temp_vol, 2), size(temp_vol, 3));
    
    for jj = 1:length(filled_locations)
        fprintf(fid, '%d\n', filled_locations(jj));
    end
    
    done(ii, length(params.model_filelist))
    
    fclose(fid);
end
