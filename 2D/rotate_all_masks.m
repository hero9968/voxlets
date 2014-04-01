% a script to rotate masks to multiple angles and save them out to
% individual image files.
%

clear
define_params
load(paths.subset_files, 'filelist')
addpath('../src')

%%
for ii = 1:length(filelist)
    
    % load in this image
    this_path = [paths.subset,  filelist(ii).name];
    this_image = imread(this_path);
    
    for jj = 1:params.n_angles
        
        % rotating image to specified angle
        this_angle = params.angles(jj);
        this_rotated_image = rotate_mask(this_image,  this_angle, params);
        
        % plot this image
        if 0
            subplot(4, 4, jj)
            imagesc(this_rotated_image)
            axis image
        end
        
        % save this rotated image
        if 1
            savename = sprintf(paths.rotated_savename, ii, jj);
            imwrite(this_rotated_image>=1, savename);
        end
        
    end
    ii
end

%%
