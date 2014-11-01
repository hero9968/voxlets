function [angles_im, conv_response, kern] = peter_angles(im, kernel_hww)


kern = genDANAKernel(kernel_hww);

% computing the conv response and getting angle
conv_response = conv2(double(im), kern, 'same');
angles_im = angle(conv_response);

angles_im(im==0) = nan;

% hack to get it to look right, in the range 0..pi
angles_im = mod(angles_im/2 + pi/2, pi);



function im = genDANAKernel(hww)

x = -hww:hww;
h = repmat(x, 2*hww+1, 1);
v = h';

im = h + v * (0 + 1j);

a = 2*angle(im);

im = cos(a) + sin(a) * (0 + 1j);

im(hww+1, hww+1) = 0;


