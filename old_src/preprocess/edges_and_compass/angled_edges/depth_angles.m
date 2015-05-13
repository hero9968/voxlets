function [final_im, conv_response, kern] = depth_angles(depth_im, edge_im, kernel_hww)

%% computing angles of the edge image
kern = genDANAKernel(kernel_hww);

% computing the conv response and getting angle
conv_response_edge = conv2(double(edge_im), kern, 'same');
angles_im = angle(conv_response_edge);
angles_im(edge_im==0) = nan;

% hack to get it to look right, in the range 0..pi
angles_im = mod(angles_im/2 + pi/2, pi);

%% getting response of the depth image from each of 8 kernels
num_kernels = 8;
conv_response = cell(1, num_kernels);
angles = linspace(pi, 0, num_kernels+1);
angles(end) = [];

for idx = 1:length(angles)
    
    % generate a kernel for this image
    temp_kern = genSimpleKernel(kernel_hww, angles(idx));
    conv_response{idx} = conv2(double(depth_im), temp_kern, 'same');
    
    % plotting code in comments below...
end

%% now discretise the edge angles
angles_im_disc = floor((angles_im/pi) * 8)+1;
%{
close all; 
h=imagesc(angles_im_disc); 
colormap(hsv); 
colorbar; 
axis image; 
set(h, 'alphadata', ~isnan(angles_im_disc))
%}

%% trying instead with sobel filter...
kern1 = genSimpleKernel(7, pi/4);
kern2 = flipud((genSimpleKernel(7, pi/2)));

respx = conv2(double(depth_im), kern1, 'same');
respy = conv2(double(depth_im), kern2, 'same');
angles = atan2(respy, respx);
%angles = mod(angle(resp1), 2*pi);

%angles(edge_im==0) = nan;
%plot_angles(angles)

%%
%temp = angles;
temp(edge_im==0) = nan;
%plot_angles(temp)

%%
kern = kustom_kernel(5);
kustom_out = conv2(double(depth_im), kern, 'same');
angles = angle(kustom_out);
angles = mod(-angles+pi, 2*pi);
subplot(121)
imagesc(angles);
axis image
colorbar
subplot(122)
imagesc(abs(kustom_out));
axis image
set(gca, 'clim', [0, 0.01])

%%
temp = angles;
temp(edge_im==0) = nan;
plot_angles(temp)

%%
mag = abs(kustom_out);
mag = mag - mean(mag(:));
logit_mag = 1./(1 + exp(-100*mag));
%imagesc(mag>-0.05)

imagesc(depth_canny(logit_mag, depth_im))
axis image
colorbar
colormap(jet)


%% now look up each edge_angle to decide which direction it should be pointing in...
final_im = nan(size(angles_im));
idx = find(~isnan(angles_im));
thresh = 0.00;
for ii = idx(:)'
    this_angle = angles_im(ii);
    this_angle_disc = angles_im_disc(ii);
    mod_yes_no = conv_response{this_angle_disc}(ii);
    if mod_yes_no < -thresh
        final_im(ii) = this_angle- pi;
    elseif mod_yes_no > thresh
        final_im(ii) = this_angle;
    else
        final_im(ii) = nan;
    end
end

% final rotation by pi!
final_im = mod(pi + final_im, 2*pi);



%{
subplot(121);
h = imagesc(angles_im)
axis image
set(h, 'alphadata', ~isnan(angles_im))
colorbar

subplot(122)
h = imagesc(final_im);
axis image
set(h, 'alphadata', ~isnan(final_im))
colorbar
colormap(hsv)
%}



function im = genSimpleKernel(hww, ang)

x = [-hww:hww];
h = repmat(x, length(x), 1);
v = h';

im = h + v * (0 + 1j);

a = angle(im);

im = cos(a+ang);% + sin(a+pi/3) * (0 + 1j);

im(hww+1, hww+1) = 0;
im = im / sum(abs(im(:)));



function im = genDANAKernel(hww)

x = -hww:hww;
h = repmat(x, 2*hww+1, 1);
v = h';

im = h + v * (0 + 1j);

a = 2*angle(im);

im = cos(a) + sin(a) * (0 + 1j);

im(hww+1, hww+1) = 0;



function im = kustom_kernel(hww)
%hww = 100;
width = 2*hww+1;
x = linspace(-pi, pi, width);
y = linspace(0, pi, width);
[X, Y] = meshgrid(x, y);

filt = sin(Y).*sin(X);
filt = filt / sum(sum(abs(filt)));
im = filt - 1j * filt';
%imagesc(filt)
%axis image
%colormap(jet

% 
%     %angles_im{idx} = angle(conv_response);
%     subplot(4, 4, 2*idx-1);
%     imagesc(temp_kern);
%     axis image
%     colorbar
%     
%     subplot(4, 4, 2*idx);
%     imagesc(conv_response{idx})
%     axis image
%     colormap(jet)
%     colorbar
%     set(gca, 'clim', [-0.1, 0.1])
%     title(num2str(angles(idx)))
