% a script to raytrace each rotated masked file and save the results to a
% file
%

clear
define_params
load(paths.subset_files, 'filelist')

addpath ../findfirst

%%
for ii = 1:length(filelist)
    
    % load in this image
    this_path = [paths.subset,  filelist(ii).name];
    this_image = imread(this_path);
    
    for jj = 1:params.n_angles
        
        % rotating image to specified angle
        rotated_name = sprintf(paths.rotated_savename, ii, jj);
        this_rotated_image = imread(rotated_name);
        
        % raytracing
        this_raytraced_depth = raytrace_2d(this_rotated_image);
        
        if 0
                    
            % plot this image
            subplot(121)
            imagesc(this_rotated_image)
            axis image
            
            subplot(122)
            plot(this_raytraced_depth)
            %set(gca,'YDir','reverse');
            set(gca, 'xlim', [0, params.im_width])
            axis equal tight
            set(gca, 'ylim', [0, params.im_height])
            
        end
        % save this rotated image
        savename = sprintf(paths.raytraced_savename, ii, jj);
        imwrite(uint16(this_raytraced_depth), savename);
        
    end
    ii
end
