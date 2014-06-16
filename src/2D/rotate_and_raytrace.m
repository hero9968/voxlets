% a script to rotate masks to multiple angles, raytrace them and save them
% to indivudual files
%

clear
cd ~/projects/shape_sharing/2D/
define_params
addpath src/utils/
addpath src/transformations/
addpath src/external/findfirst/

load(paths.filelist, 'filelist')

plotting = 0;
saving = 1;

%%
if ~exist(paths.raytraced, 'dir')
    mkdir(paths.raytraced)
end

%%
for ii = 1:length(filelist)
       
    % load in this image, and ensure if uint8 in [0, 255]
    this_path = [paths.mpeg,  filelist(ii).filename];
    img_in = uint8(imread(this_path));
    img_in(img_in > 0) = 255;
    
    % crop edges from image
    rotated = rotate_and_raytrace_mask(img_in, params.angles, params.scale);
    rotated.idx = ii;
    rotated.filename = filelist(ii).filename;
    
    % save this rotated image
    if saving
        savename = sprintf(paths.raytraced_savename, ii);
        save(savename, 'rotated');
    end
    
    done(ii, length(filelist))
end
