clear 
close all
addpath transformations/
addpath utils/

count = 100;

% loading image and plotting
im_in = imresize(rgb2gray(imread('peppers.png')), 0.4);

% defining a rotation matrix and output parameters
T = rotation_matrix(30);
width_out = 150;
height_out = 150;

% transforming my way
tic
%profile on
for ii = 1:count
    im_out{1} = myimtransfom(im_in, T, width_out, height_out);
end
%profile off viewer
my_time = toc;

%now doing the matlab way
tic
profile on
Tmat = maketform('affine', T');
for ii = 1:count
    im_out{2} = imtransform(im_in, Tmat, 'nearest', 'xdata', [1, width_out], 'ydata', [1, height_out]);
end
profile off viewer
matlab_time = toc;

%% plotting the results

subplot(141);
imagesc(im_in); 
axis image; 
set(gca, 'clim',[0. 255])


for ii = 1:2
    subplot(1,4,ii+1); 
    imagesc(im_out{ii});
    colormap(gray)
    set(gca, 'clim',[0. 255])
    axis image
end

subplot(144); 
imagesc(abs(im_out{1} - im_out{2})>2)
axis image

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

