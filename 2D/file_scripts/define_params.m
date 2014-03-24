paths.data = '../../data/';
paths.mpeg = [paths.data, '2D_shapes/MPEG7_CE-Shape-1_Part_B/'];
paths.subset = [paths.data, '2D_shapes/MPEG7_subset/'];

paths.subset_files = [paths.subset, 'filelist.mat'];

paths.rotated = [paths.data, '2D_shapes/rotated/'];
paths.rotated_savename = [paths.rotated, '%02d_%02d_mask.gif'];

paths.raytraced = [paths.data, '2D_shapes/raytraced/'];
paths.raytraced_savename = [paths.raytraced, '%02d_%02d_mask.png'];


% angles to rotate masks
params.n_angles = 16;
temp_angles = linspace(0, 360, params.n_angles+1);
params.angles = temp_angles(1:end-1);

% size of output image
params.im_height = 250;
params.im_width = 50;
params.scale = 0.2;




