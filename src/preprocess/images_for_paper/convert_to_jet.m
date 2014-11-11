function rgbImage  = convert_to_jet(m)

% converts a greyscale image to jet colourmap
m_minValue = min(m(:));
m_maxValue = max(m(:));

% Scale to 0-255;
m = 255.0 * (m - m_minValue) / (m_maxValue - m_minValue);

% Convert to 8 bit integer.
m8 = uint8(m);

% Display monochrome image.
%subplot(1,2,1);
%imshow(m8, []);
% Maximize figure.
set(gcf, 'Position', get(0,'Screensize'));

rgbImage = ind2rgb(m8, jet);