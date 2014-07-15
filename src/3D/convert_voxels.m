% script to convert the matlab voxel carving results to cleaner voxel
% outputs, in txt format to be read by a C++ program or other...
clear
close all
define_params_3d

%%
for ii = 1%:length(params.model_filelist)
    
    voxel_path = [paths.basis_models.voxelised, params.model_filelist{ii}, '.mat'];
    text_path = [paths.basis_models.voxelised_text, params.model_filelist{ii}, '.txt'];
    
    % loading data and converting to new representation
    file_contents = load(voxel_path);
    temp = single(file_contents.vol) - 40;
    transformed_vol =  1 ./ (1 + exp(-0.5 * temp));

    % writing to file
    fid = fopen(text_path, 'w');
    assert(fid~=-1, 'cannot open proper path')
    fprintf(fid, '%d, %d, %d\n', size(transformed_vol, 1), size(transformed_vol, 2), size(transformed_vol, 3));
    for jj = 1:numel(transformed_vol)
        fprintf(fid, '%0.4f\n', transformed_vol(jj));
    end
    fclose(fid);
end
