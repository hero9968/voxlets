% given a folder of images, splits them between training and test.
% will probably split at the object level so all rotated versions of a 
% shape must be in different spits
% - would also like to split at class level, but am not sure how easy this
% might be

clear
cd ~/projects/shape_sharing/2D
define_params
load(paths.filelist)
addpath src/utils
addpath src/segment
addpath src/external

%%
if ~exist(paths.segmented, 'dir')
    mkdir(paths.segmented)
end

%% setting some parameters
number_shapes = length(filelist);

for ii = 1:number_shapes
    
    % loading in the structure of raytraced depths
    readname = sprintf(paths.raytraced_savename, ii);
    load(readname, 'rotated');
    
    for jj = 1:length(rotated.rendered)
        
        this_depth = rotated.rendered(jj).depth;
        
        % segmenting the depth into a soup
        rotated.rendered(jj).segmented = ...
            segment_soup_2d(this_depth, params.segment_soup);
        
        % computing normals from the depth
        XY = [1:length(this_depth); this_depth];
        
        rotated.rendered(jj).normals = ...
            single(normals_radius_2d(XY, params.normal_radius));
        
        % converting depth to single for space reasons
        rotated.rendered(jj).depth = single(rotated.rendered(jj).depth);
        
    end
    
    % saving the whole structure back to the original location
    save(readname, 'rotated');
    done(ii, number_shapes);
end

