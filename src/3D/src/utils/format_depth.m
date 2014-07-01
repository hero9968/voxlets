function depth = format_depth(depth)
% The openGL depth renderings typically have the maximum depth represented
% as the back clipping plane. I use this function to convert them to nans. 
% In addition, this function could do other stuff if I really wanted to
% make that happen.

if ~any(isnan(depth(:)))
    max_depth = max(depth(:));
    depth(abs(depth-max_depth)<0.01) = nan;
end