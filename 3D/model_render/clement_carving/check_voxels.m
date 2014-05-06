cd ~/projects/shape_sharing/3D/model_render/clement_carving/
addpath ../../plotting/

%%
%[AA, BB] = system('python Carving.py');

%%
vox = load('temp.mat')

% some stats from the volume
sum(vox.vol(:))
sum(~vox.vol(:))
unique(vox.vol(:))

%% plotting the volume
clf
threshold = 25;%max(vox.vol(:)) - 4;
vol3d('CData', vox.vol >= threshold)
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
modelname = '11832029ed477440e279c4dee8066f27';
for ii = 1:42
    name = sprintf(modelpath, modelname, ii);
    load(name, 'depth')
    subplot(6, 7, ii)
    imagesc(depth);
    axis image
end