% a script to rotate masks to multiple angles, raytrace them and save them
% to indivudual files
%

clear
define_params
addpath src/utils/

load(paths.filelist, 'filelist')

plotting = 1;
saving = 0;

%%
if ~exist(paths.raytraced, 'dir')
    mkdir(paths.raytraced)
end

%%
profile on


for ii = 1%:length(filelist)
    
    clear rotated
    
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
profile off viewer

