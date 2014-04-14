% given a folder of images, splits them between training and test.
% will probably split at the object level so all rotated versions of a 
% shape must be in different spits
% - would also like to split at class level, but am not sure how easy this
% might be

clear
define_params
load(paths.subset_files)
addpath src/utils
addpath src/external/

if ~exist(paths.segmented, 'dir')
    mkdir(paths.segmented)
end

%% setting some parameters
number_shapes = length(filelist);

for ii = 1:number_shapes
    
    for jj = 1:params.n_angles
        
        % loading in the depth for this image
        this_filename = sprintf(paths.raytraced_savename, ii, jj);
        load(this_filename, 'this_raytraced_depth');
        
        % doing the segmentation
        segmented = segment_soup_2d(this_raytraced_depth, params);
        
        % saving the segmentation
        savename = sprintf(paths.segmented_savename, ii, jj);
        save(savename, 'segmented');

    end
    
    done(ii, number_shapes);
end



