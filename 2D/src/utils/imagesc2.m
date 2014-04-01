function h = imagesc2 ( img_data, mask_data )
% a wrapper for imagesc, with some formatting going on for nans

% sorting out scaling 0->255 to 0->1
if ndims( img_data ) == 3 && max( img_data(:) ) > 1
  img_data = double(img_data) / 255;
end

% plotting data and formatting
h = imagesc(img_data);
axis image off

% setting alpha values
if nargin == 2
    set(h, 'AlphaData', mask_data)
elseif ndims( img_data ) == 2
  set(h, 'AlphaData', ~isnan(img_data))
elseif ndims( img_data ) == 3
  set(h, 'AlphaData', ~isnan(img_data(:, :, 1)))
end

if nargout ~= 1
  clear h
end