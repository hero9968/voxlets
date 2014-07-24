function plot_matches_3d(matches)
% a function to do 3d plotting of matches into a scene
% operates on the input as given to the yaml_scene_parser, and can
% be used to verify that output

cols = 'rgbcrgbcrgbcrgbcrgbc';
filter_by_region = false;
hold on

% loop over each basis shape
for ii = 1:length(matches)

    this_vox_xyz = load_vox(matches{ii}.name);
    
    % loop over (and plot) each transformation for this basis shape
    for jj = 1:length(matches{ii}.transform)
        this_transform = [matches{ii}.transform{jj}.R, matches{ii}.transform{jj}.T'; 0 0 0 1];
        translated_match = apply_transformation_3d(this_vox_xyz, this_transform);%this_match.vox_transformation);
        
        plot3d(translated_match, cols(mod(ii+jj, length(cols))+1))
    end
end

hold off
view(-20, -84)
