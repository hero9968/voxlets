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
%profile on


for ii = 500%:length(filelist)
    
    clear rotated
    
    rotated.idx = ii;
    rotated.filename = filelist(ii).filename;
    
    % load in this image, and ensure if uint8 in [0, 255]
    this_path = [paths.mpeg,  filelist(ii).filename];
    rotated.original_image = uint8(imread(this_path));
    rotated.original_image(rotated.original_image > 0) = 255;
    
    % crop edges from image
    rotated.image = boxcrop_2d(rotated.original_image);
    
    % resize image according to predefined scale
    rotated.image = imresize(rotated.image, params.scale);
    
    % getting image height and width
    imheight = size(rotated.image, 1);
    imwidth = size(rotated.image, 2);
    
    % get the diagonal size of the image
    diag_size = sqrt(imheight^2 + imwidth^2);
    
    rotated.angles = params.angles;
    
    % loop over each possible rotation and 
    for jj = 1:length(rotated.angles)
        
        % develop the transformation matrix
        T_translate_1 = translation_matrix(-imwidth/2, -imheight/2);
        T_rotate = rotation_matrix(params.angles(jj));
        T_translate_2 = translation_matrix(2 + diag_size/2, 2 + diag_size/2);
        % ^^ adding 2 to diag size to allow for rounding errors etc.
        
        % combining and converting to MATLAB format
        T_final = T_translate_2 * T_rotate * T_translate_1;
        T_final_M = maketform('affine', T_final');
        
        % applying transformation
        this_rotated_image = imtransform((rotated.image), T_final_M, 'bilinear',...
            'xdata', [1, diag_size + 2], ...
            'ydata', [1, diag_size + 2]);
        
        % want to give function the image and the angle, and get back the
        % rotated image and a transformation which brings it back in line
        % with the origin
        
        % raytracing this rotated image
        rotated.raytraced{jj} = raytrace_2d(this_rotated_image > 128);                
        
        % plot this image
        if plotting
            [n, m] = best_subplot_dims(length(rotated.angles));
            subplot(n, m, jj)
            imagesc(this_rotated_image)
            axis image
            colormap(flipgray)
            hold on
            plot(1:length(rotated.raytraced{jj}), rotated.raytraced{jj}, 'linewidth', 3);
            hold off
        end
        
    end
    
    % save this rotated image
    if saving
        savename = sprintf(paths.raytraced_savename, ii);
        save(savename, 'rotated');
    end
    
    done(ii, length(filelist))
end
%profile off viewer

