clear 
%close all
cd ~/projects/shape_sharing/2D/src/
addpath transformations/
addpath utils/

count = 10;

% loading image and plotting
im_in = imresize(rgb2gray(imread('peppers.png')), 0.4);

% defining a rotation matrix and output parameters
T = translation_matrix(1060, 530) * rotation_matrix(200);
width_out = 1050;
height_out = 500;

% transforming my way
tic
%profile on
for ii = 1:count
    [im_out{1}, corns] = myimtransfom(im_in, T, width_out, height_out);
end
%profile off viewer
my_time = toc;

%now doing the matlab way
tic
%profile on
Tmat = maketform('affine', T');
for ii = 1:count
    [im_out{2}, xd, yd] = imtransform(im_in, Tmat, 'nearest','XYScale',1, 'xdata', [1, width_out], 'ydata', [1, height_out]);
end
%profile off viewer
matlab_time = toc;
ratio = matlab_time/my_time

% plotting the results

subplot(221);
imagesc(im_in); 
axis image; 
set(gca, 'clim',[0. 255])
title('Original image');

for ii = 1:2
    subplot(2,2,ii+1); 
    imagesc(im_out{ii});
    colormap(gray)
    set(gca, 'clim',[0. 255])
    axis image
    hold on
    plot(corns(1, :), corns(2, :), '+')
    hold off
end

subplot(222); title('My transform')
subplot(223); title('Their transform')

%
subplot(224); 
imagesc(abs(im_out{1} - im_out{2})>2)
axis image
title('Difference image')

ratio = matlab_time/my_time


%% testing my own code
T = translation_matrix(20, 60)
data_in = [1, 2, 3; 2, 4, 10];

data_out = apply_transformation_2d(data_in, T);

plot(data_in(1, :), data_in(2, :), 'b');
hold on
plot(data_out(1, :), data_out(2, :), 'r');
hold off
axis image

