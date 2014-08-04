cd ~/projects/shape_sharing/src/3D/src/voxelisation/clement_carving/
clear
addpath ../../plotting/
run ../../../define_params_3d.m

%%
%%[AA, BB] = system('python Carving.py');

%%
vox_with = load('temp_with.mat')
vox_wo = load('temp_wo.mat')

% some stats from the volume
sum(vox.vol(:))
sum(~vox.vol(:))
unique(vox.vol(:))

%% plotting the volume
subplot(121)
threshold = max(vox_with.vol(:));
vol3d('CData', vox_with.vol >= threshold - 10)
axis image
subplot(122)
threshold = max(vox_wo.vol(:));
vol3d('CData', vox_wo.vol >= threshold - 10)
axis image

%% trying a contour plot
T = +(vox.vol == max(vox.vol(:)));
S = size(vox.vol);
[X, Y, Z] = meshgrid(1:S(1), 1:S(2), 1:S(3));
Sx = linspace(1, S(1), 10);
Sy = linspace(1, S(2), 10);
Sz = linspace(1, S(3), 10);
cvals = linspace(-1,3,10);
figure
contourslice(X,Y,Z,T,Sx,Sy,Sz,cvals);
axis image

%%
modelpath = '/Users/Michael/projects/shape_sharing/data/3D/basis_models/renders/%s/depth_%d.mat';
modelname = '1f8275f1c106144ff11c3739edd52fa3';
for ii = 1:42
    name = sprintf(modelpath, modelname, ii);
    load(name, 'depth')
    depths{ii} = depth;
    subplot(6, 7, ii)
    imagesc(depth<3);
    axis image
end

%%
%modelname = '6d9b13361790d04d457ba044c28858b1';
modelname = '1f8275f1c106144ff11c3739edd52fa3';
%modelname = params.model_filelist{100};
fullpath = [paths.basis_models.voxelised '/' modelname];
T = load('temp.mat');
%T = load(fullpath);
figure
V = double(T.vol==42);
vol3d('CData', V)
axis image
view(0, 0)

%% 
for ii = 1:size(V, 3)
   imagesc(V(:, :, ii))
   drawnow
   title(num2str(ii))
   pause(0.1);
   
end

%%
xyz = reproject_depth(depths{1}, params.half_intrinsics);

xyz2 = apply_transformation_3d(xyz, 



