T = load('/Users/Michael/projects/shape_sharing/data/3D/basis_models/voxelised/1046b3d5a381a8902e5ae31c6c631a39.mat');
V = single(T.vol);
M = max(V(:));

%% generating frames
clear frame
for ii = 1:100
    frame{ii} = squeeze(V(ii, :, :)); 
end

%% visualising
for ii = 1:100
    imagesc(frame{ii});    
    axis image
    set(gca, 'clim', [0, M])
    drawnow
    pause(0.1)
    
end
