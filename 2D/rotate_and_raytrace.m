% a script to rotate masks to multiple angles, raytrace them and save them
% to indivudual files
%

clear
define_params
addpath src/utils/

load(paths.filelist, 'filelist')

plotting = 0;
saving = 1;

%%
if ~exist(paths.raytraced, 'dir')
    mkdir(paths.raytraced)
end

%%
tic
for ii = 1:length(filelist)
    
    clear rotated
    
    rotated.idx = ii;
    rotated.filename = filelist(ii).filename;
    
    % load in this image
    this_path = [paths.mpeg,  filelist(ii).filename];
    rotated.image = imread(this_path);
    
    rotated.angles = params.angles;
    
    for jj = 1:length(rotated.angles)
        
        % rotating image to specified angle
        this_angle = rotated.angles(jj);
        this_rotated_image = rotate_mask(rotated.image,  this_angle, params);
        
        % raytracing this rotated image
        rotated.raytraced{jj} = raytrace_2d(this_rotated_image);                
        
        % plot this image
        if plotting && jj <= 4 * 4
            subplot(2, 8, jj)
            imagesc(fill_grid_from_depth(rotated.raytraced{jj}, 250, 0.5))
            axis image
        end
        
    end
    
    % save this rotated image
    if saving
        savename = sprintf(paths.raytraced_savename, ii);
        save(savename, 'rotated');
    end
    
    done(ii, length(filelist))
end
toc

