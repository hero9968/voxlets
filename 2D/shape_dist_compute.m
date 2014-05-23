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
addpath src/transformations/
addpath src/segment
addpath src/external

%%
if ~exist(paths.segmented, 'dir')
    mkdir(paths.segmented)
end

%% setting some parameters
number_shapes = length(filelist);
num_samples = params.shape_dist.num_samples;

for ii = 158:number_shapes
    
    % loading in the structure of raytraced depths
    readname = sprintf(paths.raytraced_savename, ii);
    load(readname, 'rotated');
    
    for jj = 1:length(rotated.rendered)

        this_depth = rotated.rendered(jj).depth;
        this_norms = rotated.rendered(jj).normals;

        % computing normals from the depth
        XY = xy_from_depth(this_depth);
        to_remove = isnan(this_depth);
        XY(:, to_remove) = [];
        this_norms(:, to_remove) = [];
        
        rotated.rendered(jj).scale = normalise_scale(XY);
        XY_scaled = rotated.rendered(jj).scale * XY;

        % converting depth to single for space reasons
        rotated.rendered(jj).shape_dist = ...
            shape_dist_2d_dict(XY_scaled, this_norms, num_samples, params.dist_angle_dict);

        rotated.rendered(jj).image_idx = ii;
        
    end
    
    % saving the whole structure back to the original location
    save(readname, 'rotated');
    done(ii, number_shapes);
end

