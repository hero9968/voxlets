% going to check some saved files

names = {'frame_20141120T215459.461384_',   ...
'frame_20141120T134535.682741_', ...
'frame_20141120T185954.959422_', ...
'frame_20141120T213448.378796_', ...
'frame_20141120T213442.525035_', ...
'frame_20141120T213440.257049_', ...
'frame_20141120T213449.827419_'}

%%
for ii = 1:length(names)
    name = names{ii} 
    save_name = fullfile(base_path, 'mdf', [name, '.mat']);
    im = load(save_name)
    
    subplot(131)
    imagesc(im.rgb)
    axis image
    
    subplot(132)
    imagesc(im.mask)
    axis image
   
    subplot(133)
    dists = dot(im.xyz, repmat(im.up(1:3), size(im.xyz, 1), 1), 2);
    errors = abs(dists + im.up(4));
    outliers_vector = errors < 0.02; % | dists < -im.up(4);
    outliers = reshape(outliers_vector, size(im.smoothed_depth));
    imagesc(outliers)
    axis image
    drawnow 
    pause
end


%% 
for 


pause